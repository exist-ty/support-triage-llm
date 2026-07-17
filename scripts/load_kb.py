"""Считает эмбеддинги для базы знаний (src/kb.py) через локальную модель
all-minilm и загружает в kb_documents (truncate + insert) в БД triage
(pgvector, см. src/db.py::get_vector_engine)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from src.db import get_vector_engine
from src.kb import KB_DOCUMENTS
from src.ollama_client import embed
from src.rag import to_vector_literal


def load_kb() -> None:
    engine = get_vector_engine()
    contents = [doc["content"] for doc in KB_DOCUMENTS]
    embeddings = embed(contents)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE kb_documents RESTART IDENTITY CASCADE"))
        for doc, embedding in zip(KB_DOCUMENTS, embeddings):
            conn.execute(
                text(
                    "INSERT INTO kb_documents (title, content, embedding) "
                    "VALUES (:title, :content, (:embedding)::vector)"
                ),
                {"title": doc["title"], "content": doc["content"], "embedding": to_vector_literal(embedding)},
            )
    print(f"Loaded {len(KB_DOCUMENTS)} KB documents with embeddings")


if __name__ == "__main__":
    load_kb()
