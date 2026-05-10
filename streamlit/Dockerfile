FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    streamlit==1.32.0 \
    plotly==5.18.0 \
    pandas==2.1.4 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.27 \
    openai==1.14.0 \
    python-dotenv==1.0.0

COPY app.py .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
