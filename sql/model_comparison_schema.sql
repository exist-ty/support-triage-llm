-- Отдельная таблица для scripts/compare_models.py: composite PK (message_id, model)
-- позволяет держать результаты ОБЕИХ моделей по одному и тому же сообщению
-- одновременно — triage_results этого не может (message_id там PRIMARY KEY
-- сам по себе, одна модель на сообщение), а перезаписывать прод-таблицу ради
-- разового сравнения не хочется.
CREATE TABLE IF NOT EXISTS model_comparison_results (
    message_id INTEGER NOT NULL REFERENCES client_messages(message_id),
    model TEXT NOT NULL,
    category TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    priority TEXT NOT NULL,
    confidence NUMERIC(4, 3) NOT NULL,
    suggested_reply TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (message_id, model)
);
