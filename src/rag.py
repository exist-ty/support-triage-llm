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


def sparse_search(engine: Engine, query_text: str, limit: int = 10) -> list[dict]:
    """Full-text поиск через `search_tsv` (GIN-индекс, см. sql/triage_schema.sql).

    Осознанно НЕ `plainto_tsquery` — он ANDит все леммы запроса, и обращение
    клиента вида "Хочу оформить возврат..." почти никогда не матчится с
    формальной базой знаний ("Товар... можно вернуть...") только потому, что
    в документе физически нет леммы "хоч". Вместо этого леммы запроса
    получены тем же `to_tsvector('russian', ...)`, что и у документов, и
    объединены через OR (`tsquery_to_array` -> ` | `) — под капотом это то
    же ранжирование по числу совпавших термов, что и BM25-подобный поиск, а
    не строгий фильтр "все слова обязательны". Возвращает документы, уже
    отсортированные по убыванию `ts_rank` — позиция в списке используется
    как ранг в reciprocal_rank_fusion, отдельно ранг из БД не нужен."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, title, content, ts_rank(search_tsv, query_tsq) AS rank
                FROM kb_documents,
                     to_tsquery('russian', array_to_string(tsvector_to_array(to_tsvector('russian', :query_text)), ' | ')) AS query_tsq
                WHERE search_tsv @@ query_tsq
                ORDER BY rank DESC
                LIMIT :limit
                """
            ),
            {"query_text": query_text, "limit": limit},
        ).mappings().all()

    return [dict(row) for row in rows]


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009):
    score(doc) = sum по спискам 1/(k + rank). Комбинирует dense (cosine
    similarity, 0..1) и sparse (ts_rank, произвольный несравнимый масштаб)
    без нормализации скоров — берётся только позиция в каждом списке, что
    и делает RRF устойчивым к разным шкалам. k=60 — константа из
    оригинальной статьи, не подобрана под этот датасет."""
    scores: dict[int, float] = {}
    docs_by_id: dict[int, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc["id"]] = scores.get(doc["id"], 0.0) + 1.0 / (k + rank)
            docs_by_id[doc["id"]] = doc

    ordered_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    return [{**docs_by_id[doc_id], "rrf_score": scores[doc_id]} for doc_id in ordered_ids]


def hybrid_search(
    engine: Engine, query_text: str, query_embedding: list[float],
    k: int = 2, candidate_pool: int = 10,
) -> list[dict]:
    """Dense (pgvector, top_k_similar) + sparse (full-text, sparse_search),
    объединённые через reciprocal_rank_fusion. candidate_pool — сколько
    кандидатов берётся из каждой ветки перед фьюжном (больше k, иначе RRF
    выбирать не из чего)."""
    dense = top_k_similar(engine, query_embedding, k=candidate_pool)
    sparse = sparse_search(engine, query_text, limit=candidate_pool)
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    return fused[:k]
