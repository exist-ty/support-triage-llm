"""Retrieval поверх kb_documents: brute-force cosine similarity в numpy.
Оправдано только при малом размере базы знаний — см. комментарий в
sql/triage_schema.sql про отказ от pgvector."""
from __future__ import annotations

import numpy as np


def top_k_similar(
    query_embedding: list[float],
    documents: list[dict],
    k: int = 2,
) -> list[dict]:
    """documents: [{"id", "title", "content", "embedding"}, ...].

    Возвращает k документов с наибольшим cosine similarity к query_embedding.
    """
    query = np.array(query_embedding)
    query_norm = query / np.linalg.norm(query)

    scored = []
    for doc in documents:
        vec = np.array(doc["embedding"])
        vec_norm = vec / np.linalg.norm(vec)
        similarity = float(np.dot(query_norm, vec_norm))
        scored.append((similarity, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:k]]
