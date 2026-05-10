"""
Bronze layer: ingest raw JSON messages from Kafka and persist to MinIO as Parquet.

Uses Structured Streaming with trigger(availableNow=True) so the job processes
all messages accumulated since the last checkpoint and then exits cleanly —
ideal for Airflow-scheduled batch-streaming runs.
"""

import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType, TimestampType,
)

load_dotenv()

# ── Environment ───────────────────────────────────────────────
KAFKA_SERVERS   = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "real-estate-listings")
MINIO_ENDPOINT  = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS    = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET    = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET    = os.getenv("MINIO_BUCKET", "lakehouse")

BRONZE_PATH     = f"s3a://{MINIO_BUCKET}/bronze/listings"
CHECKPOINT_PATH = f"s3a://{MINIO_BUCKET}/checkpoints/bronze"

# ── Expected schema of JSON messages produced by producer.py ──
# Adjust field names to match your actual CSV columns.
MESSAGE_SCHEMA = StructType([
    StructField("price",           DoubleType(),    True),
    StructField("area",            DoubleType(),    True),
    StructField("rooms",           IntegerType(),   True),
    StructField("floor",           IntegerType(),   True),
    StructField("total_floors",    IntegerType(),   True),
    StructField("district",        StringType(),    True),
    StructField("address",         StringType(),    True),
    StructField("building_type",   StringType(),    True),
    StructField("published_at",    StringType(),    True),
    StructField("current_timestamp", StringType(),  True),  # added by producer
])


def build_spark() -> SparkSession:
    """Create SparkSession wired to MinIO via S3A and Kafka connector."""
    return (
        SparkSession.builder
        .appName("BronzeLayer")
        # Packages: Kafka connector + S3A filesystem + AWS SDK
        .config(
            "spark.jars.packages",
            ",".join([
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
                "org.apache.hadoop:hadoop-aws:3.3.4",
                "com.amazonaws:aws-java-sdk-bundle:1.12.262",
            ]),
        )
        # S3A → MinIO configuration
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
    """Read from Kafka, parse JSON, write raw Parquet to MinIO bronze layer."""
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # Read Kafka stream — value column contains the raw JSON bytes
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Cast binary value → string, then parse JSON into struct columns
    parsed = (
        raw_stream
        .select(
            from_json(col("value").cast("string"), MESSAGE_SCHEMA).alias("data"),
            col("timestamp").alias("kafka_timestamp"),
        )
        .select("data.*", "kafka_timestamp")
        # Add a processing timestamp for lineage tracking
        .withColumn("bronze_loaded_at", current_timestamp())
    )

    # Write to MinIO; trigger=availableNow processes backlog then stops
    query = (
        parsed.writeStream
        .format("parquet")
        .option("path", BRONZE_PATH)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(availableNow=True)
        .start()
    )

    query.awaitTermination()
    print(f"[bronze] Finished writing to {BRONZE_PATH}")


if __name__ == "__main__":
    main()
