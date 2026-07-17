# Support Triage LLM — триаж обращений клиентов через локальную LLM

![Tests](https://github.com/exist-ty/support-triage-llm/actions/workflows/test.yml/badge.svg)

Пет-проект на стыке Data/LLM-инженерии: обращения клиентов интернет-магазина
классифицируются локальной LLM (Qwen2.5-3B-Instruct через Ollama, без единого
внешнего API-вызова и без затрат на токены) с RAG-контекстом из базы знаний
магазина, результат пишется в Postgres и связывается с той же БД, что
использует [`etl-portfolio`](../etl-portfolio) и
[`product-marketing-analytics`](../product-marketing-analytics).

Три вопроса, на которые отвечает этот репозиторий:

1. Можно ли классифицировать и приоритизировать обращения в поддержку локальной
   3B-моделью без GPU и без внешнего API — и где проходит граница её надёжности?
2. Различается ли профиль обращений (доля негатива/high-priority) между
   маркетинговыми каналами привлечения?
3. Как выглядит production-грейд RAG (векторный индекс в БД, а не Python-цикл)
   и как оценить качество LLM-классификации метриками, а не на глаз?

## Стек

Python, Ollama (Qwen2.5-3B-Instruct + all-minilm), PostgreSQL + pgvector
(HNSW), pydantic, scikit-learn (evaluation), Docker, pytest.

## Архитектура

```
client_messages (БД triage/pgvector; customer_id — из реального
                 customer_id в stg_customers, БД etl_portfolio)
        │
        ▼
  embed (all-minilm, Ollama)
        │
        ▼
  top-k векторный поиск в Postgres (pgvector, HNSW, оператор <=>)  ──▶  src/rag.py
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
        │
        ▼
  scripts/evaluate_llm.py: category vs. true_category → F1, confusion matrix
```

## Структура

- `sql/triage_schema.sql` — `kb_documents` (`VECTOR(384)` + HNSW-индекс) /
  `client_messages` (+ `true_category`) / `triage_results` — БД `triage`
  (см. «Production RAG: pgvector» ниже)
- `src/kb.py` — база знаний магазина (доставка, возврат, гарантия, оплата и т.д.)
- `src/ollama_client.py` — тонкий клиент поверх Ollama HTTP API (chat + embed)
- `src/rag.py` — retrieval: векторный поиск в Postgres (`<=>`, pgvector)
- `src/triage.py` — промпт, pydantic-схема ответа, retry-логика
- `src/db.py` — два движка: основная БД `etl_portfolio` (только чтение
  `stg_customers`/`stg_orders`) и БД `triage` (pgvector)
- `scripts/load_kb.py` — считает эмбеддинги базы знаний и грузит в `triage`
- `scripts/generate_messages.py` — синтетические обращения, привязанные к
  реальным `customer_id`/заказам из `etl-portfolio`, с `true_category`
  (тема шаблона-источника — не ручная разметка, но известная "истина")
- `scripts/run_triage.py` — основной пайплайн (резюмируемый: пропускает уже
  обработанные сообщения)
- `scripts/channel_triage_summary.py` — кросс-БД связка с `stg_customers.channel`
  (JOIN теперь в pandas — см. «Production RAG» ниже)
- `scripts/evaluate_llm.py` — F1 по классам и confusion matrix (`true_category`
  vs. предсказание модели)
- `tests/` — pytest на парсинг/валидацию структурированного вывода LLM (без
  реальных вызовов модели — на моках)
- `docker-compose.yml` + `Dockerfile` — Ollama + Postgres/pgvector + приложение
  одной командой

## Как запустить

Локально:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5:3b-instruct
ollama pull all-minilm
docker compose up -d vector-db          # Postgres + pgvector, порт 5433
psql -U postgres -h localhost -p 5433 -d triage -f sql/triage_schema.sql
python scripts/load_kb.py
python scripts/generate_messages.py
python scripts/run_triage.py
python scripts/channel_triage_summary.py
python scripts/evaluate_llm.py
pytest
```

Через Docker (поднимает Ollama + Postgres/pgvector-контейнеры + прогоняет
весь пайплайн одной командой):
```
docker compose up --build
```

## Production RAG: pgvector

Раньше: эмбеддинги как `double precision[]`, cosine similarity — brute-force
Python-цикл в numpy (`src/rag.py`), оправдано только при базе знаний в
10-15 документов (в локальной установке PostgreSQL 17 на Windows extension
`vector` недоступен без сборки из исходников — не числится даже в
`pg_available_extensions`).

Теперь: `kb_documents.embedding VECTOR(384)` в отдельном Postgres-контейнере
`pgvector/pgvector:pg17` (порт 5433, БД `triage`) — образ уже содержит
собранное расширение, компилировать самому не нужно. HNSW-индекс
(`vector_cosine_ops`) вместо IVFFlat: строится быстрее и точнее на
базах такого масштаба (IVFFlat выигрывает только когда HNSW не влезает по
памяти при построении на очень больших коллекциях). `src/rag.py` теперь
не выкачивает все документы в Python — `ORDER BY embedding <=> :query LIMIT k`
выполняется движком БД с использованием индекса.

**Плата за это:** `client_messages.customer_id` больше не `FOREIGN KEY` на
`stg_customers.customer_id` — Postgres не умеет внешние ключи между базами
(тем более между контейнерами). Ссылочная целостность держится на том, что
`generate_messages.py` берёт `customer_id` из реального `SELECT` по
`etl_portfolio`, а не constraint'ом. По той же причине
`scripts/channel_triage_summary.py` больше не может JOIN'ить
`triage_results` и `stg_customers` одним SQL-запросом — две разные БД,
поэтому объединение делает `pandas.merge` в Python. Осознанный компромисс,
не забытый баг.

## Результаты на реальных данных

45 синтетических обращений (11 шаблонов-тем, честно сгенерированы с
привязкой к реальным заказам), полностью прогнаны через Qwen2.5-3B-Instruct
на CPU (GPU — NVIDIA MX250 2GB, собранный под неё Ollama-рантайм падает с
несовместимостью CUDA-тулчейна, см. «Честные ограничения» ниже).

| предсказанная category | count |
|---|---|
| return | 15 |
| wholesale | 6 |
| delivery_status | 6 |
| payment | 6 |
| loyalty | 4 |
| complaint_rude | 4 |
| exchange | 2 |
| praise | 1 |
| defect | 1 |

Средняя уверенность модели (`confidence`) — 0.83, среднее время обработки
одного сообщения (embed + retrieval + классификация) — ~32 сек. на этом CPU.

**В часах, не только в секундах.** Если оператор поддержки тратит на прочтение,
классификацию и черновик ответа условно ~2 минуты на обращение (оценка, не
измерение — реального оператора в этом проекте нет), а автоматический первый
проход занимает ~32 сек., это ~73% экономии времени именно на этом шаге. На
объёме 1 000 обращений в месяц — это ≈27 часов операторского времени. Это
честная оценка выигрыша **первого прохода**, а не полной автоматизации: при
accuracy 69% (см. ниже) результат — черновик с категорией и приоритетом,
который оператор всё ещё проверяет, особенно на классах, где модель
систематически ошибается (`cancel`, `off_topic`) — экономия там, где модель
уверена и права, а не универсальная замена оператора.

**Пример триажа high-priority/negative обращения:**
> «Очень недоволен обслуживанием по заказу #1780, оператор грубо ответил в
> чате. Разбираюсь третий день без результата!»

→ `category=complaint_rude, sentiment=negative, priority=high`, черновик
ответа модель сгенерировала со ссылкой на конкретную проблему из обращения.

**Каналы** (`scripts/channel_triage_summary.py`, кросс-БД `pandas.merge` с
`stg_customers` из etl-portfolio): на этой выборке `referral` даёт
наибольшую долю негатива/high-priority обращений (25%) против 10-12.5% у
остальных каналов — но при n=45 это не более чем наблюдение, не
статистически значимый вывод. Совпадение `negative_share`/`high_priority_share`
по каждому каналу — не баг агрегации: на этой выборке модель ни разу не
поставила `priority=high` без `sentiment=negative` и наоборот.

**Кросс-репо наблюдение.** В [`product-marketing-analytics`](../product-marketing-analytics)
`referral` — канал с лучшим ROMI (см. его README). Здесь же у `referral`
наибольшая доля недовольных обращений. Оба вывода честно ограничены малым n
(45 обращений / 200 клиентов), но направление одно и то же в двух независимо
посчитанных выборках: дешёвое привлечение через `referral` стоит проверять не
только по стоимости (CAC/ROMI), но и по качеству клиентского опыта, прежде
чем масштабировать канал.

## Оценка качества (`scripts/evaluate_llm.py`)

`true_category` — тема шаблона, из которого сгенерировано сообщение (см.
`generate_messages.py`): не ручная разметка человеком, а точно известная
по построению данных "истина". Сравнение с реальным предсказанием модели:

```
accuracy: 0.69   macro F1: 0.64   weighted F1: 0.61   (n=45)
```

| category | precision | recall | F1 | support |
|---|---|---|---|---|
| complaint_rude | 1.00 | 1.00 | 1.00 | 4 |
| defect | 1.00 | 1.00 | 1.00 | 1 |
| exchange | 1.00 | 1.00 | 1.00 | 2 |
| loyalty | 1.00 | 1.00 | 1.00 | 4 |
| payment | 1.00 | 1.00 | 1.00 | 6 |
| return | 0.60 | 1.00 | 0.75 | 9 |
| delivery_status | 0.33 | 1.00 | 0.50 | 2 |
| wholesale | 0.33 | 1.00 | 0.50 | 2 |
| praise | 1.00 | 0.20 | 0.33 | 5 |
| **cancel** | 0.00 | 0.00 | 0.00 | 6 |
| **off_topic** | 0.00 | 0.00 | 0.00 | 4 |

![Confusion matrix](exports/confusion_matrix.png)

Confusion matrix называет ровно то, что раньше было честной, но качественной
формулировкой ("категории схлопываются") — теперь с числами и конкретным
направлением ошибки:

- **Все 6 `cancel` → `return`.** Модель видит "хочу отменить заказ" и "хочу
  вернуть заказ" как один и тот же интент — оба про "не хочу этот заказ",
  различие в моменте (до/после доставки) 3B-модель на CPU без fine-tuning
  не удерживает без явного примера в промпте.
- **Все 4 `off_topic` → `wholesale`.** Сообщение про сотрудничество с
  блогерами (не про политики магазина) RAG находит ближе всего к документу
  "Корпоративные и оптовые заказы" — модель классифицирует по
  retrieved-контексту, а не по факту нерелевантности контекста вопросу.
- **4 из 5 `praise` → `delivery_status`.** Хвалебные сообщения в шаблоне
  содержат "Спасибо за быструю доставку" — модель цепляется за слово
  "доставка", а не за общий позитивный тон обращения.

## Честные ограничения

- **Категории схлопываются predictable-образом** — см. confusion matrix
  выше: не случайный шум, а систематическая путаница у семантически близких
  категорий (cancel/return) и переоценка релевантности RAG-контекста
  (off_topic/wholesale).
- **n=45.** Честный end-to-end прогон на слабом железе за разумное время,
  но 1-2 support на класс (`defect`, `exchange`, `praise`) — F1 на таких
  классах шумит от одного сообщения к другому, не статистическая оценка.
- **Железо.** GPU (NVIDIA MX250, 2GB VRAM) не тянет CUDA-тулчейн текущей
  сборки Ollama-рантайма (`CUDA error: the provided PTX was compiled with an
  unsupported toolchain` → крэш llama-server) — весь инференс идёт на CPU
  (`OLLAMA_LLM_LIBRARY=cpu`, `CUDA_VISIBLE_DEVICES=""`), ~32 сек/сообщение.
- **8GB RAM.** Запуск Docker Desktop параллельно с CPU-инференсом модели
  реально приводил к падению Ollama-сервера при нехватке памяти — на этой
  машине это не гипотетический, а наблюдавшийся риск.

## Связь с другими репозиториями

`client_messages.customer_id` — реальный `customer_id` из `stg_customers`
([`etl-portfolio`](../etl-portfolio)), `scripts/channel_triage_summary.py`
джойнит результат триажа с `channel` оттуда же — без дублирования данных
между репозиториями. Аналитика по ROMI/LTV/retention — в
[`product-marketing-analytics`](../product-marketing-analytics).
