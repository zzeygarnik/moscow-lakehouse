"""
Lakehouse pipeline DAG: Producer → Bronze → Silver → Gold.

Schedule: every 5 minutes.
Execution model:
  - run_producer : BashOperator — runs producer.py inside the Airflow container.
  - run_bronze/silver/gold : SparkSubmitOperator — submits PySpark scripts to
    the Spark cluster in CLIENT MODE (driver on Airflow, executors on workers).
    All env vars (MinIO, Kafka, PG credentials) are inherited from the Airflow
    container environment and passed through SparkConf inside each script.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# ── Default task arguments ────────────────────────────────────
default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
    "email_on_failure": False,
}

# ── Spark packages shared across all Spark tasks ──────────────
# These are resolved by the driver at submission time (Maven/Ivy cache).
_PACKAGES = ",".join([
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
    "org.apache.hadoop:hadoop-aws:3.3.4",
    "com.amazonaws:aws-java-sdk-bundle:1.12.262",
    "org.postgresql:postgresql:42.7.1",
])

# ── DAG definition ─────────────────────────────────────────────
with DAG(
    dag_id="lakehouse_pipeline",
    description="Moscow real estate: streaming ingest → Bronze → Silver → Gold",
    default_args=default_args,
    schedule_interval="*/5 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,          # prevent overlapping pipeline runs
    tags=["lakehouse", "real-estate", "moscow"],
) as dag:

    # ── Task 1: Emit new listings from CSV to Kafka ────────────
    # Runs producer.py on the Airflow container; all env vars are
    # inherited from the container (KAFKA_BOOTSTRAP_SERVERS, etc.)
    run_producer = BashOperator(
        task_id="run_producer",
        bash_command="python /opt/airflow/producer.py",
        # Give Kafka a moment to receive all messages before Spark reads
        append_env=True,
    )

    # ── Task 2: Bronze — Kafka → MinIO (raw Parquet) ───────────
    run_bronze = SparkSubmitOperator(
        task_id="run_bronze",
        application="/opt/airflow/spark/jobs/bronze.py",
        conn_id="spark_default",        # AIRFLOW_CONN_SPARK_DEFAULT in env
        packages=_PACKAGES,
        verbose=False,
        name="bronze_layer_{{ ds_nodash }}_{{ ts_nodash }}",
    )

    # ── Task 3: Silver — bronze Parquet → cleaned Parquet ──────
    run_silver = SparkSubmitOperator(
        task_id="run_silver",
        application="/opt/airflow/spark/jobs/silver.py",
        conn_id="spark_default",
        packages=_PACKAGES,
        verbose=False,
        name="silver_layer_{{ ds_nodash }}_{{ ts_nodash }}",
    )

    # ── Task 4: Gold — silver → PostgreSQL data marts ──────────
    run_gold = SparkSubmitOperator(
        task_id="run_gold",
        application="/opt/airflow/spark/jobs/gold.py",
        conn_id="spark_default",
        packages=_PACKAGES,
        verbose=False,
        name="gold_layer_{{ ds_nodash }}_{{ ts_nodash }}",
    )

    # ── Pipeline chain ─────────────────────────────────────────
    run_producer >> run_bronze >> run_silver >> run_gold
