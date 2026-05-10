"""
Kafka producer for Moscow real estate listings.

Reads a local CSV file, samples 10% of rows, and emits each row
as a JSON message to a Kafka topic — simulating a real-time stream
of new property listings.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ── Config ────────────────────────────────────────────────────
load_dotenv()

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC             = os.getenv("KAFKA_TOPIC", "real-estate-listings")
CSV_PATH          = os.getenv("CSV_PATH", "data/raw_dataset.csv")
SAMPLE_FRACTION   = float(os.getenv("SAMPLE_FRACTION", "0.1"))
EMIT_DELAY_SEC    = float(os.getenv("EMIT_DELAY_SEC", "0.1"))   # pause between messages
LOG_EVERY_N       = int(os.getenv("LOG_EVERY_N", "50"))         # log progress every N messages

KAFKA_RETRY_ATTEMPTS = 10
KAFKA_RETRY_DELAY    = 6   # seconds between connection attempts

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────

def _create_producer() -> KafkaProducer:
    """
    Try to connect to Kafka with retries.

    Raises RuntimeError if all attempts are exhausted.
    """
    for attempt in range(1, KAFKA_RETRY_ATTEMPTS + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                acks="all",             # wait for leader + replicas to confirm
                retries=3,
                linger_ms=5,            # small batching window for throughput
            )
            log.info("Connected to Kafka at %s (attempt %d/%d)",
                     BOOTSTRAP_SERVERS, attempt, KAFKA_RETRY_ATTEMPTS)
            return producer
        except NoBrokersAvailable:
            log.warning(
                "Kafka not ready yet (attempt %d/%d). Retrying in %ds …",
                attempt, KAFKA_RETRY_ATTEMPTS, KAFKA_RETRY_DELAY,
            )
            time.sleep(KAFKA_RETRY_DELAY)

    raise RuntimeError(
        f"Could not connect to Kafka at {BOOTSTRAP_SERVERS} "
        f"after {KAFKA_RETRY_ATTEMPTS} attempts."
    )


def _load_sample(path: str, fraction: float) -> pd.DataFrame:
    """Load CSV and return a random sample of `fraction` rows."""
    log.info("Reading dataset from %s", path)
    df = pd.read_csv(path)
    sample = df.sample(frac=fraction, random_state=None)
    log.info("Dataset size: %d rows | Sample size: %d rows (%.0f%%)",
             len(df), len(sample), fraction * 100)
    return sample


def _on_send_error(exc: Exception) -> None:
    """Async callback for failed deliveries."""
    log.error("Message delivery failed: %s", exc)


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    """Entry point: sample CSV → stream every row to Kafka as JSON."""
    producer = _create_producer()
    sample   = _load_sample(CSV_PATH, SAMPLE_FRACTION)

    log.info("Starting stream → topic '%s'", TOPIC)
    total = len(sample)

    for idx, (_, row) in enumerate(sample.iterrows(), start=1):
        # Convert row to plain dict; NaN → None for valid JSON
        record = {
            k: (None if pd.isna(v) else v)
            for k, v in row.to_dict().items()
        }
        # Attach ingestion timestamp (ISO-8601, UTC)
        record["current_timestamp"] = datetime.now(timezone.utc).isoformat()

        producer.send(TOPIC, value=record).add_errback(_on_send_error)

        if idx % LOG_EVERY_N == 0 or idx == total:
            log.info("Sent %d / %d messages to '%s'", idx, total, TOPIC)

        time.sleep(EMIT_DELAY_SEC)

    producer.flush()
    log.info("Stream complete. %d messages delivered.", total)


if __name__ == "__main__":
    main()
