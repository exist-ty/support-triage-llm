# Support Triage LLM — триаж обращений клиентов через локальную LLM

Пет-проект на стыке Data/LLM-инженерии: обращения клиентов интернет-магазина
классифицируются локальной LLM (Qwen2.5-3B-Instruct через Ollama, без единого
внешнего API-вызова и без затрат на токены) с RAG-контекстом из базы знаний
магазина, результат пишется в Postgres и связывается с той же БД, что
использует [`etl-portfolio`](../etl-portfolio) и
[`product-marketing-analytics`](../product-marketing-analytics).

Два вопроса, на которые отвечает этот репозиторий:

1. Можно ли классифицировать и приоритизировать обращения в поддержку локальной
   3B-моделью без GPU и без внешнего API — и где проходит граница её надёжности?
2. Различается ли профиль обращений (доля негатива/high-priority) между
   маркетинговыми каналами привлечения?

## Стек

Python, Ollama (Qwen2.5-3B-Instruct + all-minilm), PostgreSQL, pydantic,
Docker, pytest.

## Архитектура

```
client_messages (Postgres, привязаны к реальным customer_id из etl-portfolio)
        │
        ▼
  embed (all-minilm, Ollama)
        │
        ▼
  top-k cosine similarity против kb_documents  ──▶  src/rag.py
        │
        ▼
  промпт: обращение + top-k выдержек из базы знаний
        │
        ▼
  Qwen2.5-3B-Instruct (Ollama, format=json)
        │
        ▼
  pydantic-валидация + retry на невалидный JSON  ──▶  src/triage.py
        │
        ▼
  triage_results (category, sentiment, priority, confidence, suggested_reply)
```

## Структура

- `sql/triage_schema.sql` — `kb_documents` / `client_messages` / `triage_results`
- `src/kb.py` — база знаний магазина (доставка, возврат, гарантия, оплата и т.д.)
- `src/ollama_client.py` — тонкий клиент поверх Ollama HTTP API (chat + embed)
- `src/rag.py` — retrieval: cosine similarity на numpy (без pgvector, см. ниже)
- `src/triage.py` — промпт, pydantic-схема ответа, retry-логика
- `scripts/load_kb.py` — считает эмбеддинги базы знаний и грузит в БД
- `scripts/generate_messages.py` — синтетические обращения, привязанные к
  реальным `customer_id`/заказам из `etl-portfolio`
- `scripts/run_triage.py` — основной пайплайн (резюмируемый: пропускает уже
  обработанные сообщения)
- `scripts/channel_triage_summary.py` — JOIN с `stg_customers.channel`
- `tests/` — pytest на парсинг/валидацию структурированного вывода LLM (без
  реальных вызовов модели — на моках)
- `docker-compose.yml` + `Dockerfile` — Ollama + приложение одной командой

## Как запустить

Локально:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5:3b-instruct
ollama pull all-minilm
psql -U postgres -d etl_portfolio -f sql/triage_schema.sql
python scripts/load_kb.py
python scripts/generate_messages.py
python scripts/run_triage.py
python scripts/channel_triage_summary.py
pytest
```

Через Docker (поднимает Ollama-контейнер + прогоняет весь пайплайн одной
командой):
```
docker compose up --build
```

## Почему не pgvector

Расширение `vector` недоступно в этой локальной установке PostgreSQL 17 (не
числится даже в `pg_available_extensions` — потребовалась бы сборка из
исходников на Windows). При базе знаний в 10 документов ставить pgvector ради
этого не оправдано: эмбеддинги хранятся как обычный `double precision[]`, а
cosine similarity считается brute-force на стороне Python (numpy,
`src/rag.py`) — решение по объёму данных, аналогичное индексам в
`etl-portfolio`. При росте базы знаний до тысяч документов это первое, что
стоит поменять.

## Результаты на реальных данных

45 синтетических обращений (11 шаблонов-тем, честно сгенерированы с
привязкой к реальным заказам), полностью прогнаны через Qwen2.5-3B-Instruct
на CPU (GPU — NVIDIA MX250 2GB, собранный под неё Ollama-рантайм падает с
несовместимостью CUDA-тулчейна, см. «Железо и ограничения» ниже).

| category | count |
|---|---|
| return | 15 |
| loyalty | 8 |
| payment | 6 |
| delivery_status | 4 |
| complaint_rude | 4 |
| praise | 3 |
| exchange | 2 |
| wholesale | 2 |
| defect | 1 |

Средняя уверенность модели (`confidence`) — 0.83, среднее время обработки
одного сообщения (embed + retrieval + классификация) — ~43 сек. на этом CPU.

**Пример триажа high-priority/negative обращения:**
> «Очень недоволен обслуживанием по заказу #939, оператор грубо ответил в
> чате. Разбираюсь третий день без результата!»

→ `category=complaint_rude, sentiment=negative, priority=high`, черновик
ответа модель сгенерировала со ссылкой на конкретный номер заказа.

**Каналы** (`scripts/channel_triage_summary.py`, JOIN с `stg_customers` из
etl-portfolio): на этой выборке `context_ads` даёт наибольшую долю
негатива и high-priority обращений (20%) против ~10-12% у остальных
каналов — но при n=45 это не более чем наблюдение, не статистически
значимый вывод.

## Честные ограничения

- **Категории схлопываются.** Из 11 тем-шаблонов в исходных данных (включая
  `cancel` и `off_topic`) в реальных результатах классификации ни разу не
  встретились `cancel` и `off_topic` — 3B-модель на CPU без fine-tuning
  сводит более редкие/двусмысленные обращения к соседним категориям
  (`return`, `payment`). Для 11-класс­овой классификации 3B-параметров на
  границе достаточности — это ожидаемо, но не гарантировано без валидации
  на размеченных данных.
- **n=45.** Достаточно, чтобы честно прогнать пайплайн end-to-end на слабом
  железе за разумное время, недостаточно для статистических выводов о
  каналах или точной оценке accuracy классификатора.
- **Железо.** GPU (NVIDIA MX250, 2GB VRAM) не тянет CUDA-тулчейн текущей
  сборки Ollama-рантайма (`CUDA error: the provided PTX was compiled with an
  unsupported toolchain` → крэш llama-server) — весь инференс идёт на CPU
  (`OLLAMA_LLM_LIBRARY=cpu`, `CUDA_VISIBLE_DEVICES=""`), ~40 сек/сообщение.
- **8GB RAM.** Запуск Docker Desktop параллельно с CPU-инференсом модели
  реально приводил к падению Ollama-сервера при нехватке памяти — на этой
  машине это не гипотетический, а наблюдавшийся риск.

## Связь с другими репозиториями

`client_messages.customer_id` — реальный `customer_id` из `stg_customers`
([`etl-portfolio`](../etl-portfolio)), `scripts/channel_triage_summary.py`
джойнит результат триажа с `channel` оттуда же — без дублирования данных
между репозиториями. Аналитика по ROMI/LTV/retention — в
[`product-marketing-analytics`](../product-marketing-analytics).
