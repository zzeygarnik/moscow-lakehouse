"""
Moscow Real Estate Analytics — Streamlit Dashboard + AI Chat Agent.

Dashboard: visualises Gold-layer data marts queried directly from PostgreSQL.
AI Agent:  Two-step Text-to-SQL pipeline powered by DeepSeek API (OpenAI-compatible).
           Step 1 — DeepSeek generates SQL from the user's natural-language question.
           Step 2 — Python executes the SQL and sends results back to DeepSeek
                    for a short human-readable answer in Russian.
"""

import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from sqlalchemy import create_engine, text

# ── Environment ───────────────────────────────────────────────
# Loaded from .env via docker-compose env_file; dotenv not needed here.
PG_HOST   = os.getenv("POSTGRES_HOST",     "postgres")
PG_PORT   = os.getenv("POSTGRES_PORT",     "5432")
PG_DB     = os.getenv("POSTGRES_DB",       "gold_db")
PG_USER   = os.getenv("POSTGRES_USER",     "admin")
PG_PASS   = os.getenv("POSTGRES_PASSWORD", "changeme")

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL    = "deepseek-chat"

DB_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# ── Gold-layer DDL embedded in the system prompt ──────────────
# Mirrors the tables created by gold.py exactly.
_DB_SCHEMA = """
CREATE TABLE mart_avg_price_by_district (
    district          VARCHAR,
    avg_price         DOUBLE PRECISION,    -- average listing price in RUB
    avg_price_per_sqm DOUBLE PRECISION,    -- average price per square meter in RUB
    listing_count     BIGINT
);

CREATE TABLE mart_listings_by_day (
    listing_date      DATE,
    listings_count    BIGINT,
    avg_price         DOUBLE PRECISION,
    avg_price_per_sqm DOUBLE PRECISION
);

CREATE TABLE mart_avg_price_by_rooms (
    rooms             INTEGER,
    avg_price         DOUBLE PRECISION,
    avg_price_per_sqm DOUBLE PRECISION,
    avg_area_sqm      DOUBLE PRECISION,
    listing_count     BIGINT
);
"""

_SQL_SYSTEM_PROMPT = f"""You are a PostgreSQL expert. \
The database contains Moscow real estate analytics.

Schema:
{_DB_SCHEMA}

Rules:
- Return ONLY a valid PostgreSQL SQL query. No markdown. No code fences. No explanation.
- Use only the tables and columns defined in the schema above.
- Prices are in Russian rubles (RUB). Areas in square meters (m²).
- If the question cannot be answered from this schema, return exactly:
  SELECT 'Cannot answer from available data' AS message;
"""

_ANSWER_SYSTEM_PROMPT = """You are a concise Moscow real estate analyst assistant.
The user asked a question in Russian.
A SQL query was run and returned data shown below.
Write a short, clear answer in Russian (under 5 sentences).
Be specific: reference the actual numbers from the data."""


# ── Cached resources (created once per Streamlit server lifetime) ──
@st.cache_resource
def get_engine():
    """Shared SQLAlchemy engine — one connection pool for all requests."""
    return create_engine(DB_URL, pool_pre_ping=True)


@st.cache_resource
def get_deepseek_client() -> OpenAI:
    """Shared DeepSeek client (OpenAI-compatible endpoint)."""
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# ── Data loaders (cached 5 min — matches pipeline schedule) ──────
@st.cache_data(ttl=300)
def load_district_data() -> pd.DataFrame:
    q = """
        SELECT district, avg_price, avg_price_per_sqm, listing_count
        FROM   mart_avg_price_by_district
        ORDER  BY avg_price DESC
        LIMIT  15
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(q), conn)


@st.cache_data(ttl=300)
def load_daily_data() -> pd.DataFrame:
    q = """
        SELECT listing_date, listings_count, avg_price
        FROM   mart_listings_by_day
        ORDER  BY listing_date
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(q), conn)


@st.cache_data(ttl=300)
def load_rooms_data() -> pd.DataFrame:
    q = """
        SELECT rooms, avg_price, avg_price_per_sqm, listing_count
        FROM   mart_avg_price_by_rooms
        ORDER  BY rooms
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(q), conn)


# ── Text-to-SQL pipeline ──────────────────────────────────────
def _strip_fences(raw: str) -> str:
    """Remove markdown code fences that the model may add despite instructions."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "",    raw)
    return raw.strip()


def generate_sql(client: OpenAI, question: str) -> str:
    """Step 1 — natural language → PostgreSQL query via DeepSeek."""
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _SQL_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        temperature=0,
        max_tokens=512,
    )
    return _strip_fences(resp.choices[0].message.content or "")


def execute_sql(sql: str) -> tuple[pd.DataFrame | None, str | None]:
    """Step 2 — run SQL in PostgreSQL; return (dataframe, error_or_None)."""
    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df, None
    except Exception as exc:
        return None, str(exc)


def generate_answer(client: OpenAI, question: str, sql: str, df: pd.DataFrame) -> str:
    """Step 3 — data → human-readable Russian answer via DeepSeek."""
    data_str = df.head(20).to_string(index=False)
    payload = (
        f"User question: {question}\n\n"
        f"SQL executed:\n{sql}\n\n"
        f"Result ({len(df)} rows total):\n{data_str}"
    )
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
            {"role": "user",   "content": payload},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return resp.choices[0].message.content or "No answer generated."


# ── Dashboard section ─────────────────────────────────────────
def render_dashboard() -> None:
    st.header("📊 Рынок недвижимости Москвы — Обзор")

    try:
        district_df = load_district_data()
        daily_df    = load_daily_data()
        rooms_df    = load_rooms_data()
        data_ok = True
    except Exception as exc:
        st.warning(f"Данные дашборда недоступны: {exc}")
        st.info("Запустите Airflow-пайплайн хотя бы раз, чтобы заполнить Gold-таблицы.")
        data_ok = False

    if not data_ok:
        return

    # Ряд 1: два графика
    col1, col2 = st.columns(2)

    with col1:
        if not district_df.empty:
            fig = px.bar(
                district_df,
                x="district",
                y="avg_price",
                color="avg_price_per_sqm",
                color_continuous_scale="Blues",
                title="Средняя цена по районам (Топ-15)",
                labels={
                    "avg_price":         "Средняя цена (₽)",
                    "avg_price_per_sqm": "₽/м²",
                    "district":          "Район",
                },
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Данных по районам пока нет.")

    with col2:
        if not daily_df.empty:
            fig2 = px.line(
                daily_df,
                x="listing_date",
                y="listings_count",
                title="Количество новых объявлений по дням",
                labels={"listing_date": "Дата", "listings_count": "Объявлений"},
                markers=True,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Данных по дням пока нет.")

    # Ряд 2: график по комнатам + таблица по районам
    col3, col4 = st.columns(2)

    with col3:
        if not rooms_df.empty:
            fig3 = px.bar(
                rooms_df,
                x="rooms",
                y="avg_price",
                title="Средняя цена по количеству комнат",
                labels={"rooms": "Комнат", "avg_price": "Средняя цена (₽)"},
                color="avg_price",
                color_continuous_scale="Oranges",
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if not district_df.empty:
            st.subheader("Сводка по районам")
            display_df = district_df.rename(columns={
                "district":          "Район",
                "avg_price":         "Средняя цена",
                "avg_price_per_sqm": "Цена за м²",
                "listing_count":     "Объявлений",
            })
            st.dataframe(
                display_df.style.format({
                    "Средняя цена": "{:,.0f} ₽",
                    "Цена за м²":   "{:,.0f} ₽/м²",
                }),
                use_container_width=True,
                height=300,
            )


# ── Chat section ──────────────────────────────────────────────
def render_chat() -> None:
    st.header("🤖 ИИ-агент — Задайте вопрос")
    st.caption(
        "Примеры: «Где самые дорогие двушки?»  •  «Топ-5 районов по цене за м²»  •  «Сколько объявлений было вчера?»"
    )

    if not DEEPSEEK_API_KEY:
        st.error("⚠️ DEEPSEEK_API_KEY не задан. Добавьте его в .env и перезапустите.")
        return

    client = get_deepseek_client()

    if "messages" not in st.session_state:
        st.session_state.messages: list[dict] = []

    # Replay existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("🔍 Сгенерированный SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
            if msg.get("data") is not None and not msg["data"].empty:
                with st.expander(f"📋 Данные ({len(msg['data'])} строк)", expanded=False):
                    st.dataframe(msg["data"], use_container_width=True)

    # New user input
    if user_input := st.chat_input("Задайте вопрос на русском языке..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            # Шаг 1: генерация SQL
            with st.spinner("Шаг 1/3 — Генерация SQL…"):
                sql = generate_sql(client, user_input)

            # Шаг 2: выполнение SQL
            with st.spinner("Шаг 2/3 — Выполнение запроса…"):
                df, error = execute_sql(sql)

            if error:
                err_text = (
                    f"⚠️ **Ошибка выполнения SQL.**\n\n"
                    f"**Запрос:**\n```sql\n{sql}\n```\n\n"
                    f"**Ошибка:**\n```\n{error}\n```"
                )
                st.markdown(err_text)
                st.session_state.messages.append({
                    "role": "assistant", "content": err_text,
                    "sql": sql, "data": None,
                })
                return

            # Шаг 3: формирование ответа
            with st.spinner("Шаг 3/3 — Формирование ответа…"):
                answer = generate_answer(client, user_input, sql, df)

            st.markdown(answer)
            with st.expander("🔍 Сгенерированный SQL", expanded=False):
                st.code(sql, language="sql")
            with st.expander(f"📋 Данные ({len(df)} строк)", expanded=False):
                st.dataframe(df, use_container_width=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sql": sql,
                "data": df,
            })


# ── Entry point ───────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Аналитика недвижимости Москвы",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_dashboard()
    st.divider()
    render_chat()


if __name__ == "__main__":
    main()
