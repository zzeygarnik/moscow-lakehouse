"""
Silver layer: clean, validate, and enrich Bronze Parquet data.

Reads all Parquet from the bronze layer, applies quality rules,
casts types, and computes derived columns. Writes to the silver
layer in Append mode so each Airflow run adds only new partitions.

Deduplication key: (district, address, price, area, published_at).
Critical null filter: rows without price OR area are dropped.
"""

import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, round as spark_round, to_timestamp, current_timestamp,
)
from pyspark.sql.types import DoubleType, IntegerType

load_dotenv()

# ── Environment ───────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT",  "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET   = os.getenv("MINIO_BUCKET",     "lakehouse")

BRONZE_PATH = f"s3a://{MINIO_BUCKET}/bronze/listings"
SILVER_PATH = f"s3a://{MINIO_BUCKET}/silver/listings"

# Timestamp format used in published_at field (adjust to match your CSV)
TS_FORMAT = "yyyy-MM-dd HH:mm:ss"

# Columns that must be non-null for a row to be considered valid
CRITICAL_COLS = ["price", "area"]

# Deduplication subset — uniquely identifies a listing
DEDUP_SUBSET = ["district", "address", "price", "area", "published_at"]


def build_spark() -> SparkSession:
    """Create SparkSession with S3A/MinIO configuration."""
    return (
        SparkSession.builder
        .appName("SilverLayer")
        .config(
            "spark.jars.packages",
            ",".join([
                "org.apache.hadoop:hadoop-aws:3.3.4",
                "com.amazonaws:aws-java-sdk-bundle:1.12.262",
            ]),
        )
        .config("spark.hadoop.fs.s3a.endpoint",          MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key",        MINIO_ACCESS)
        .config("spark.hadoop.fs.s3a.secret.key",        MINIO_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate()
    )


def main() -> None:
    """Run the Silver transformation pipeline."""
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Read Bronze ────────────────────────────────────────
    df = spark.read.parquet(BRONZE_PATH)
    raw_count = df.count()
    print(f"[silver] Bronze rows read: {raw_count}")

    # ── 2. Drop duplicates ────────────────────────────────────
    df = df.dropDuplicates(DEDUP_SUBSET)
    print(f"[silver] After dedup: {df.count()}")

    # ── 3. Drop rows with critical nulls ──────────────────────
    df = df.dropna(subset=CRITICAL_COLS)
    print(f"[silver] After null filter: {df.count()}")

    # ── 4. Cast & normalise types ─────────────────────────────
    df = (
        df
        # Numeric columns — enforce correct types after JSON parse
        .withColumn("price",       col("price").cast(DoubleType()))
        .withColumn("area",        col("area").cast(DoubleType()))
        .withColumn("rooms",       col("rooms").cast(IntegerType()))
        .withColumn("floor",       col("floor").cast(IntegerType()))
        .withColumn("total_floors",col("total_floors").cast(IntegerType()))
        # Parse published_at string → proper timestamp
        .withColumn("published_at", to_timestamp(col("published_at"), TS_FORMAT))
    )

    # ── 5. Business derivation: price per square meter ────────
    df = df.withColumn(
        "price_per_sqm",
        spark_round(col("price") / col("area"), 2),
    )

    # ── 6. Lineage column ─────────────────────────────────────
    df = df.withColumn("silver_loaded_at", current_timestamp())

    # ── 7. Write to MinIO silver layer ───────────────────────
    # Append mode: each Airflow run adds new data without reprocessing history.
    # Partition by district for efficient downstream reads.
    (
        df.write
        .format("parquet")
        .mode("append")
        .partitionBy("district")
        .save(SILVER_PATH)
    )

    print(f"[silver] Wrote {df.count()} rows to {SILVER_PATH}")


if __name__ == "__main__":
    main()
