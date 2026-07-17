"""Retrieval поверх kb_documents: векторный поиск средствами Postgres
(pgvector, HNSW-индекс, оператор <=> — cosine distance), а не brute-force
в numpy. Раньше здесь был Python-цикл по всем документам (оправдано только
при базе знаний в 10-15 документов, см. историю sql/triage_schema.sql) —
теперь similarity считает сам движок БД, как в проде."""
from __future__ import annotations

from sqlalchemy import Engine, text


def to_vector_literal(embedding: list[float]) -> str:
    """Текстовое представление вектора для psycopg2 (`'[...]'::vector`) —
    переиспользуется в scripts/load_kb.py при записи эмбеддингов."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def top_k_similar(engine: Engine, query_embedding: list[float], k: int = 2) -> list[dict]:
    """Возвращает k документов с наибольшим cosine similarity к query_embedding.

    `embedding <=> :query` — cosine distance (0 = идентичны, 2 = противоположны);
    ORDER BY по нему же использует HNSW-индекс. similarity = 1 - distance,
    чтобы сохранить ту же семантику (больше = ближе), что была в старой
    numpy-версии.
    """
    query_vector = to_vector_literal(query_embedding)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, title, content, 1 - (embedding <=> (:query)::vector) AS similarity
                FROM kb_documents
                ORDER BY embedding <=> (:query)::vector
                LIMIT :k
                """
            ),
            {"query": query_vector, "k": k},
        ).mappings().all()

    return [dict(row) for row in rows]
