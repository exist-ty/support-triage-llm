"""Основной пайплайн: для каждого необработанного client_messages —
embed -> hybrid retrieval топ-k из kb_documents (dense pgvector + sparse
full-text, объединённые RRF — см. src/rag.py::hybrid_search) ->
классификация через Qwen -> запись в triage_results. Обрабатывает
сообщения по одному (не батчами): это пакетный ночной джоб, а не
realtime-сервис, задержка в секундах на сообщение здесь не проблема."""
import json
import os
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_vector_engine
from src.ollama_client import CHAT_MODEL, EMBED_MODEL, embed
from src.rag import hybrid_search
from src.triage import classify_message

if sys.platform == "win32":
    # mlflow печатает emoji ("🏃 View run...") при завершении run — в стандартной
    # cp1252-консоли Windows это падает с UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")

TOP_K = 2
RRF_K = 60  # см. src/rag.py::reciprocal_rank_fusion — константа из статьи, не тюнится этим скриптом

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5501")
MLFLOW_EXPERIMENT = "support-triage"
# run_id пишется сюда, чтобы evaluate_llm.py дологировал метрики (accuracy,
# F1) в ТОТ ЖЕ run — метрики появляются только после того, как triage
# готов, а не в момент классификации отдельных сообщений
RUN_ID_PATH = PROJECT_ROOT / "exports" / "mlflow_run_id.json"


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

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="run_triage") as active_run:
        mlflow.log_param("chat_model", CHAT_MODEL)
        mlflow.log_param("embed_model", EMBED_MODEL)
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("rrf_k", RRF_K)

        processed = 0
        latencies_ms = []
        for _, row in pending.iterrows():
            start = time.perf_counter()

            query_embedding = embed([row["message_text"]])[0]
            retrieved = hybrid_search(engine, row["message_text"], query_embedding, k=TOP_K)
            result = classify_message(row["message_text"], retrieved)

            latency_ms = int((time.perf_counter() - start) * 1000)
            latencies_ms.append(latency_ms)

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

        mlflow.log_metric("messages_processed", processed)
        if latencies_ms:
            mlflow.log_metric("avg_latency_ms", sum(latencies_ms) / len(latencies_ms))

        RUN_ID_PATH.parent.mkdir(exist_ok=True)
        RUN_ID_PATH.write_text(json.dumps({"run_id": active_run.info.run_id}))

    print(f"Done: processed {processed} messages")


if __name__ == "__main__":
    run()
