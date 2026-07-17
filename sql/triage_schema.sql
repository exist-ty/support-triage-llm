-- Применяется к ОТДЕЛЬНОЙ базе triage (контейнер pgvector/pgvector:pg17,
-- порт 5433 — см. docker-compose.yml и src/db.py::get_vector_engine),
-- а не к etl_portfolio, где живут stg_customers/stg_orders. Локальный
-- Postgres 17 на Windows не даёт поставить extension vector без сборки из
-- исходников (не числится в pg_available_extensions) — pgvector/pgvector
-- даёт его "из коробки", это и есть переход на pgvector.
--
-- Плата за это: client_messages.customer_id больше не FK на
-- stg_customers.customer_id — Postgres не умеет внешние ключи между базами
-- (тем более между разными контейнерами). Ссылочная целостность здесь не
-- гарантируется constraint'ом, а держится на том, что generate_messages.py
-- берёт customer_id из реального SELECT по etl_portfolio. Кросс-базовый
-- JOIN (channel_triage_summary.py) поэтому теперь делается в pandas,
-- а не в SQL — см. комментарий в этом скрипте.
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS triage_results CASCADE;
DROP TABLE IF EXISTS client_messages CASCADE;
DROP TABLE IF EXISTS kb_documents CASCADE;

-- all-minilm (Ollama) отдаёt 384-мерные эмбеддинги
CREATE TABLE kb_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    -- GENERATED ALWAYS ... STORED: пересчитывается автоматически при
    -- INSERT/UPDATE, load_kb.py не должен сам его поддерживать в актуальном
    -- состоянии. 'russian'-конфигурация — стемминг под язык базы знаний
    -- (src/kb.py), без неё full-text не находил бы "доставка" по "доставки".
    search_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('russian', title || ' ' || content)) STORED
);

-- HNSW — быстрее строится и точнее IVFFlat на малых/средних базах знаний
-- (IVFFlat нужен только когда HNSW не влезает по памяти при построении);
-- vector_cosine_ops — под тот же cosine similarity, что был в src/rag.py
CREATE INDEX idx_kb_documents_embedding_hnsw
    ON kb_documents USING hnsw (embedding vector_cosine_ops);

-- GIN — стандартный индекс под full-text (см. src/rag.py::sparse_search,
-- «Hybrid search» в README)
CREATE INDEX idx_kb_documents_search_tsv
    ON kb_documents USING gin (search_tsv);

CREATE TABLE client_messages (
    message_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,  -- см. комментарий сверху: не FK, другая БД
    message_text TEXT NOT NULL,
    -- Истинная категория из шаблона генерации (scripts/generate_messages.py)
    -- — не ручная разметка человеком, а известная "истина" по построению
    -- синтетических данных. Используется для scripts/evaluate_llm.py.
    true_category TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_client_messages_created_at ON client_messages(created_at);

CREATE TABLE triage_results (
    message_id INTEGER PRIMARY KEY REFERENCES client_messages(message_id),
    category TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    priority TEXT NOT NULL,
    confidence NUMERIC(4, 3) NOT NULL,
    suggested_reply TEXT NOT NULL,
    retrieved_doc_ids INTEGER[] NOT NULL,
    model TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_triage_results_category ON triage_results(category);
CREATE INDEX idx_triage_results_priority ON triage_results(priority);
