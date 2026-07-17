-- Таблицы триажа живут в той же БД etl_portfolio, что и etl-portfolio/
-- product-marketing-analytics: customer_id/channel переиспользуются из
-- stg_customers, а не дублируются.
--
-- pgvector в локальной установке PostgreSQL 17 недоступен (не в
-- pg_available_extensions — потребовалась бы сборка из исходников на
-- Windows). При базе знаний в 10-15 документов это не оправдано: эмбеддинги
-- хранятся как double precision[], а cosine similarity считается на
-- стороне Python (numpy) в src/rag.py. Явный компромисс по объёму данных,
-- как и решение про индексы в etl-portfolio.
DROP TABLE IF EXISTS triage_results CASCADE;
DROP TABLE IF EXISTS client_messages CASCADE;
DROP TABLE IF EXISTS kb_documents CASCADE;

CREATE TABLE kb_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL
);

CREATE TABLE client_messages (
    message_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES stg_customers(customer_id),
    message_text TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

-- Маркетинговый канал не дублируется здесь: он уже есть в
-- stg_customers.channel и достаётся через JOIN по customer_id (см.
-- scripts/channel_triage_summary.py) — дублирование только увеличило бы
-- риск рассинхронизации без выигрыша в этом объёме данных.
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
