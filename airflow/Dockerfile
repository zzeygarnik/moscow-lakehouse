FROM apache/airflow:2.8.1

USER root

# Install Java 17 JRE — required by spark-submit client
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download Spark 3.5.0 distribution (provides spark-submit binary for client mode)
ARG SPARK_VERSION=3.5.0
ENV SPARK_HOME=/opt/spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${SPARK_HOME}/bin:${JAVA_HOME}/bin:${PATH}"

RUN curl -fsSL \
    "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
    | tar -xz -C /opt \
    && mv "/opt/spark-${SPARK_VERSION}-bin-hadoop3" "${SPARK_HOME}" \
    && chown -R airflow:root "${SPARK_HOME}"

USER airflow

# Airflow Spark provider + producer script dependencies
RUN pip install --no-cache-dir \
    apache-airflow-providers-apache-spark==4.7.1 \
    kafka-python==2.0.2 \
    pandas==2.1.4 \
    python-dotenv==1.0.0
