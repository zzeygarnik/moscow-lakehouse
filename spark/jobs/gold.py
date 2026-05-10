"""
Gold layer: build aggregated data marts from Silver and write to PostgreSQL.

Three marts are produced on every run:
  1. mart_avg_price_by_district  — average price and price/sqm per district
  2. mart_listings_by_day        — daily listing volume and average price trend
  3. mart_avg_price_by_rooms     — average price segmented by room count

All tables are written in OVERWRITE mode so the Gold layer always reflects
the full Silver dataset — safe for aggregations that must be recalculated
from scratch on each pipeline run.
"""

import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    avg, col, count, round as spark_round, to_date,
)

load_dotenv()

# ── Environment ───────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT",  "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET   = os.getenv("MINIO_BUCKET",     "lakehouse")

PG_HOST   = os.getenv("POSTGRES_HOST",     "postgres")
PG_PORT   = os.getenv("POSTGRES_PORT",     "5432")
PG_DB     = os.getenv("POSTGRES_DB",       "gold_db")
PG_USER   = os.getenv("POSTGRES_USER",     "admin")
PG_PASS   = os.getenv("POSTGRES_PASSWORD", "changeme")

SILVER_PATH = f"s3a://{MINIO_BUCKET}/silver/listings"
JDBC_URL    = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
JDBC_PROPS  = {
    "user":     PG_USER,
    "password": PG_PASS,
    "driver":   "org.postgresql.Driver",
}


def build_spark() -> SparkSession:
    """Create SparkSession with S3A and PostgreSQL JDBC support."""
    return (
        SparkSession.builder
        .appName("GoldLayer")
        .config(
            "spark.jars.packages",
            ",".join([
                "org.apache.hadoop:hadoop-aws:3.3.4",
                "com.amazonaws:aws-java-sdk-bundle:1.12.262",
                "org.postgresql:postgresql:42.7.1",
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


def write_to_postgres(df: DataFrame, table: str) -> None:
    """Overwrite a PostgreSQL table with the given DataFrame."""
    (
        df.write
        .jdbc(
            url=JDBC_URL,
            table=table,
            mode="overwrite",
            properties=JDBC_PROPS,
        )
    )
    print(f"[gold] Written {df.count()} rows → postgres:{table}")


def main() -> None:
    """Compute Gold aggregations and persist to PostgreSQL."""
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── Read Silver ───────────────────────────────────────────
    silver = spark.read.parquet(SILVER_PATH)
    print(f"[gold] Silver rows loaded: {silver.count()}")

    # ── Mart 1: Average price & price/sqm per district ────────
    mart_district = (
        silver
        .groupBy("district")
        .agg(
            spark_round(avg("price"), 2).alias("avg_price"),
            spark_round(avg("price_per_sqm"), 2).alias("avg_price_per_sqm"),
            count("*").alias("listing_count"),
        )
        .orderBy(col("avg_price").desc())
    )
    write_to_postgres(mart_district, "mart_avg_price_by_district")

    # ── Mart 2: Daily listing volume & average price trend ─────
    mart_daily = (
        silver
        .withColumn("listing_date", to_date(col("published_at")))
        .groupBy("listing_date")
        .agg(
            count("*").alias("listings_count"),
            spark_round(avg("price"), 2).alias("avg_price"),
            spark_round(avg("price_per_sqm"), 2).alias("avg_price_per_sqm"),
        )
        .orderBy("listing_date")
    )
    write_to_postgres(mart_daily, "mart_listings_by_day")

    # ── Mart 3: Average price segmented by room count ─────────
    mart_rooms = (
        silver
        .filter(col("rooms").isNotNull() & (col("rooms") > 0))
        .groupBy("rooms")
        .agg(
            spark_round(avg("price"), 2).alias("avg_price"),
            spark_round(avg("price_per_sqm"), 2).alias("avg_price_per_sqm"),
            spark_round(avg("area"), 2).alias("avg_area_sqm"),
            count("*").alias("listing_count"),
        )
        .orderBy("rooms")
    )
    write_to_postgres(mart_rooms, "mart_avg_price_by_rooms")

    print("[gold] All marts written successfully.")


if __name__ == "__main__":
    main()
