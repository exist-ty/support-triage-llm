import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_engine() -> Engine:
    """Основная БД etl_portfolio: stg_customers/stg_orders (только читаем)."""
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', '')}"
        f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'etl_portfolio')}"
    )


def get_vector_engine() -> Engine:
    """БД triage (pgvector/pgvector, отдельный контейнер): kb_documents с
    vector-эмбеддингами, client_messages, triage_results."""
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('VECTOR_DB_USER', 'postgres')}:{os.getenv('VECTOR_DB_PASSWORD', '')}"
        f"@{os.getenv('VECTOR_DB_HOST', 'localhost')}:{os.getenv('VECTOR_DB_PORT', '5433')}/{os.getenv('VECTOR_DB_NAME', 'triage')}"
    )
