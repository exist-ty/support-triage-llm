"""Честная оценка качества триажа: сравнивает true_category (истинная тема
шаблона, из которого сгенерировано сообщение — см. generate_messages.py) с
category, которую реально предсказала Qwen2.5-3B-Instruct. Не ручная
разметка человеком, а известная по построению синтетических данных
"истина" — но это ровно та же метрика (F1 по классам, confusion matrix),
что применяется при разметке людьми."""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_vector_engine

if sys.platform == "win32":
    # mlflow печатает emoji ("🏃 View run...") при завершении run — в стандартной
    # cp1252-консоли Windows это падает с UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5501")
MLFLOW_EXPERIMENT = "support-triage"
# записан run_triage.py — дологируем метрики в тот же run, чтобы параметры
# (модель, top_k, rrf_k) и метрики качества были в одном месте в MLflow UI
RUN_ID_PATH = PROJECT_ROOT / "exports" / "mlflow_run_id.json"

QUERY = """
SELECT m.true_category, r.category AS predicted_category
FROM client_messages m
JOIN triage_results r ON r.message_id = m.message_id
WHERE m.true_category IS NOT NULL
"""


def plot_confusion_matrix(labels: list[str], matrix, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(0.6 * len(labels) + 3, 0.6 * len(labels) + 2))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказанная категория")
    ax.set_ylabel("Истинная категория (true_category)")
    ax.set_title("Confusion matrix: триаж Qwen2.5-3B-Instruct")

    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value:
                ax.text(j, i, str(value), ha="center", va="center",
                        color="white" if value > threshold else "black", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8, label="сообщений")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def evaluate() -> None:
    df = pd.read_sql(QUERY, get_vector_engine())
    if df.empty:
        raise RuntimeError("Нет размеченных сообщений с true_category — прогнать generate_messages.py заново")

    labels = sorted(set(df["true_category"]) | set(df["predicted_category"]))

    print(f"n = {len(df)}\n")
    print(classification_report(df["true_category"], df["predicted_category"], labels=labels, zero_division=0))

    matrix = confusion_matrix(df["true_category"], df["predicted_category"], labels=labels)
    accuracy = (df["true_category"] == df["predicted_category"]).mean()
    f1_macro = f1_score(df["true_category"], df["predicted_category"], labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(df["true_category"], df["predicted_category"], labels=labels, average="weighted", zero_division=0)

    export_dir = PROJECT_ROOT / "exports"
    export_dir.mkdir(exist_ok=True)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(export_dir / "confusion_matrix.csv")
    plot_confusion_matrix(labels, matrix, export_dir / "confusion_matrix.png")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_id = json.loads(RUN_ID_PATH.read_text())["run_id"] if RUN_ID_PATH.exists() else None

    with mlflow.start_run(run_id=run_id, run_name=None if run_id else "evaluate_llm"):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_artifact(str(export_dir / "confusion_matrix.png"))
        mlflow.log_artifact(str(export_dir / "confusion_matrix.csv"))


if __name__ == "__main__":
    evaluate()
