"""Честное сравнение локальной Qwen2.5-3B-Instruct (Ollama) и облачной
Llama 3.3 70B Instruct (Groq API) на ОДНОМ И ТОМ ЖЕ наборе сообщений с
известной true_category (см. scripts/generate_messages.py). Retrieval
(hybrid_search) не меняется между моделями — сравнивается только шаг
классификации, чтобы разница была честно объяснима моделью, а не разным
контекстом.

Данные: message_text — синтетические сообщения, сгенерированные локально,
не переписка реальных клиентов. Для Llama 3.3 70B это означает, что текст
уходит по HTTPS на инфраструктуру Groq — здесь это не утечка PII, но если
этот backend когда-нибудь укажут на реальные обращения, это решение нужно
согласовать отдельно (см. README, раздел "Локально vs облако").

Результат пишется в model_comparison_results (sql/model_comparison_schema.sql)
— отдельная таблица с PRIMARY KEY (message_id, model), чтобы держать обе
модели по каждому сообщению одновременно; triage_results (продовая таблица)
не трогается."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_vector_engine
from src.llama_groq_client import GROQ_CHAT_MODEL
from src.llama_groq_client import chat_json as groq_chat_json
from src.ollama_client import CHAT_MODEL as OLLAMA_CHAT_MODEL
from src.ollama_client import chat_json as ollama_chat_json
from src.ollama_client import embed
from src.rag import hybrid_search
from src.triage import classify_message

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

TOP_K = 2
BACKENDS = [
    (OLLAMA_CHAT_MODEL, ollama_chat_json),
    (GROQ_CHAT_MODEL, groq_chat_json),
]


def fetch_labeled_messages(engine) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT message_id, message_text, true_category FROM client_messages "
        "WHERE true_category IS NOT NULL ORDER BY message_id",
        engine,
    )


def run_comparison() -> None:
    engine = get_vector_engine()
    messages = fetch_labeled_messages(engine)
    if messages.empty:
        raise RuntimeError("Нет размеченных сообщений — прогнать scripts/generate_messages.py")

    with engine.begin() as conn:
        conn.execute(text(open(PROJECT_ROOT / "sql" / "model_comparison_schema.sql", encoding="utf-8").read()))

    print(f"n = {len(messages)} сообщений, модели: {', '.join(m for m, _ in BACKENDS)}\n")

    for model_name, chat_fn in BACKENDS:
        with engine.connect() as conn:
            done_ids = set(
                conn.execute(
                    text("SELECT message_id FROM model_comparison_results WHERE model = :model"),
                    {"model": model_name},
                ).scalars()
            )
        # Возобновляемость: облачный backend рационирован (реально наблюдался
        # 429 на ~33-м из 45 запросов) — перезапуск после сбоя не должен
        # повторно тратить квоту на уже посчитанные message_id.
        pending = messages[~messages["message_id"].isin(done_ids)]
        if done_ids:
            print(f"--- {model_name} (пропускаю {len(done_ids)} уже посчитанных) ---")
        else:
            print(f"--- {model_name} ---")

        for i, row in pending.iterrows():
            query_embedding = embed([row["message_text"]])[0]
            retrieved = hybrid_search(engine, row["message_text"], query_embedding, k=TOP_K)

            start = time.perf_counter()
            result = classify_message(row["message_text"], retrieved, chat_fn=chat_fn, model_name=model_name)
            latency_ms = int((time.perf_counter() - start) * 1000)

            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO model_comparison_results
                            (message_id, model, category, sentiment, priority,
                             confidence, suggested_reply, latency_ms)
                        VALUES
                            (:message_id, :model, :category, :sentiment, :priority,
                             :confidence, :suggested_reply, :latency_ms)
                        ON CONFLICT (message_id, model) DO UPDATE SET
                            category = EXCLUDED.category, sentiment = EXCLUDED.sentiment,
                            priority = EXCLUDED.priority, confidence = EXCLUDED.confidence,
                            suggested_reply = EXCLUDED.suggested_reply, latency_ms = EXCLUDED.latency_ms,
                            created_at = now()
                        """
                    ),
                    {
                        "message_id": int(row["message_id"]),
                        "model": model_name,
                        "category": result.category,
                        "sentiment": result.sentiment,
                        "priority": result.priority,
                        "confidence": result.confidence,
                        "suggested_reply": result.suggested_reply,
                        "latency_ms": latency_ms,
                    },
                )
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(messages)}")

    report_and_plot(engine, messages)


def report_and_plot(engine, messages: pd.DataFrame) -> None:
    results = pd.read_sql("SELECT message_id, model, category, latency_ms FROM model_comparison_results", engine)
    merged = results.merge(messages[["message_id", "true_category"]], on="message_id")

    rows = []
    for model_name, _ in BACKENDS:
        subset = merged[merged["model"] == model_name]
        accuracy = (subset["true_category"] == subset["category"]).mean()
        f1_macro = f1_score(subset["true_category"], subset["category"], average="macro", zero_division=0)
        rows.append(
            {
                "model": model_name,
                "n": len(subset),
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "avg_latency_ms": subset["latency_ms"].mean(),
                "p95_latency_ms": subset["latency_ms"].quantile(0.95),
            }
        )

    report = pd.DataFrame(rows)
    print("\n" + report.to_string(index=False))

    export_dir = PROJECT_ROOT / "exports"
    export_dir.mkdir(exist_ok=True)
    report.to_csv(export_dir / "model_comparison.csv", index=False)
    plot_comparison(report, export_dir / "model_comparison.png")


def plot_comparison(report: pd.DataFrame, output_path: Path) -> None:
    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(10, 4))

    ax_acc.bar(report["model"], report["accuracy"], color="#3987e5")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Точность классификации")
    ax_acc.tick_params(axis="x", rotation=20)

    ax_lat.bar(report["model"], report["avg_latency_ms"], color="#1baf7a")
    ax_lat.set_ylabel("мс")
    ax_lat.set_title("Средняя задержка на сообщение")
    ax_lat.tick_params(axis="x", rotation=20)

    fig.suptitle("Qwen2.5-3B (локально, Ollama) vs Llama 3.3 70B (облако, Groq)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    run_comparison()
