"""Связывает результат триажа с маркетинговым каналом клиента —
triage_results/client_messages живут в БД triage (pgvector, отдельный
контейнер), stg_customers — в etl_portfolio (etl-portfolio). Раз это две
разные базы Postgres, кросс-базовый JOIN здесь делает pandas.merge, а не
SQL: показывает, различается ли профиль обращений (доля жалоб/приоритет)
между каналами привлечения — мостик к product-marketing-analytics, а не
дублирование его логики."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_engine, get_vector_engine

TRIAGE_QUERY = """
SELECT m.customer_id, r.category, r.sentiment, r.priority
FROM triage_results r
JOIN client_messages m ON m.message_id = r.message_id
"""


def build_summary() -> pd.DataFrame:
    triage = pd.read_sql(TRIAGE_QUERY, get_vector_engine())
    customers = pd.read_sql("SELECT customer_id, channel FROM stg_customers", get_engine())

    df = triage.merge(customers, on="customer_id", how="inner")

    summary = (
        df.groupby("channel")
        .agg(
            messages=("category", "count"),
            negative_share=("sentiment", lambda s: round((s == "negative").mean(), 3)),
            high_priority_share=("priority", lambda s: round((s == "high").mean(), 3)),
        )
        .reset_index()
        .sort_values("negative_share", ascending=False)
    )
    return summary


if __name__ == "__main__":
    summary = build_summary()
    export_dir = PROJECT_ROOT / "exports"
    export_dir.mkdir(exist_ok=True)
    summary.to_csv(export_dir / "channel_triage_summary.csv", index=False)
    print(summary.to_string(index=False))
