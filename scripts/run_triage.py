"""Основной пайплайн: для каждого необработанного client_messages —
embed -> hybrid retrieval топ-k из kb_documents (dense pgvector + sparse
full-text, объединённые RRF — см. src/rag.py::hybrid_search) ->
классификация через Qwen -> запись в triage_results. Обрабатывает
сообщения по одному (не батчами): это пакетный ночной джоб, а не
realtime-сервис, задержка в секундах на сообщение здесь не проблема."""
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_vector_engine
from src.ollama_client import CHAT_MODEL, embed
from src.rag import hybrid_search
from src.triage import classify_message

TOP_K = 2


def fetch_pending_messages(engine) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT m.message_id, m.message_text
        FROM client_messages m
        LEFT JOIN triage_results r ON r.message_id = m.message_id
        WHERE r.message_id IS NULL
        ORDER BY m.message_id
        """,
        engine,
    )


def run() -> None:
    engine = get_vector_engine()
    pending = fetch_pending_messages(engine)
    print(f"Processing {len(pending)} messages with {CHAT_MODEL}...")

    processed = 0
    for _, row in pending.iterrows():
        start = time.perf_counter()

        query_embedding = embed([row["message_text"]])[0]
        retrieved = hybrid_search(engine, row["message_text"], query_embedding, k=TOP_K)
        result = classify_message(row["message_text"], retrieved)

        latency_ms = int((time.perf_counter() - start) * 1000)

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO triage_results
                        (message_id, category, sentiment, priority, confidence,
                         suggested_reply, retrieved_doc_ids, model, latency_ms)
                    VALUES
                        (:message_id, :category, :sentiment, :priority, :confidence,
                         :suggested_reply, :retrieved_doc_ids, :model, :latency_ms)
                    """
                ),
                {
                    "message_id": int(row["message_id"]),
                    "category": result.category,
                    "sentiment": result.sentiment,
                    "priority": result.priority,
                    "confidence": result.confidence,
                    "suggested_reply": result.suggested_reply,
                    "retrieved_doc_ids": [doc["id"] for doc in retrieved],
                    "model": CHAT_MODEL,
                    "latency_ms": latency_ms,
                },
            )
        processed += 1
        if processed % 20 == 0:
            print(f"  {processed}/{len(pending)}")

    print(f"Done: processed {processed} messages")


if __name__ == "__main__":
    run()
