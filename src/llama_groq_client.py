"""Тонкий клиент поверх Llama 3.3 70B Instruct, хостится Groq (OpenAI-совместимый
REST API). Тот же интерфейс, что у ollama_client.py (chat_json(prompt, system)),
чтобы triage.py мог переключать backend без изменения кода классификации.

Важно про данные: в отличие от ollama_client.py (модель крутится локально,
ничего не покидает машину), здесь message_text целиком уходит по HTTPS на
инфраструктуру Groq. В этом проекте client_messages — синтетические сообщения
(scripts/generate_messages.py), не данные реальных клиентов, поэтому это не
утечка PII. Если этот backend когда-то указывать на реальные обращения
клиентов, это решение нужно сначала явно согласовать (см. README, раздел
"Локально vs облако")."""
from __future__ import annotations

import os
import time

import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Реально наблюдалось на этом ключе: TPM-лимит (x-ratelimit-limit-tokens)
# 12000/мин, и цикл по 45 сообщениям с RAG-контекстом в промпте выбивает 429
# уже на ~33-м запросе — не гипотетический edge case, а то, что случилось
# при первом прогоне scripts/compare_models.py.
MAX_RETRIES_ON_RATE_LIMIT = 5


def _retry_after_seconds(response: requests.Response) -> float:
    """Groq отдаёт Retry-After на 429 не всегда — берём x-ratelimit-reset-tokens
    как запасной вариант (у него бывает суффикс 'ms'/'s', см. заголовки живого
    ответа), иначе фиксированная пауза с запасом."""
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    reset = response.headers.get("x-ratelimit-reset-tokens") or response.headers.get("x-ratelimit-reset-requests")
    if reset:
        try:
            if reset.endswith("ms"):
                return float(reset[:-2]) / 1000
            if reset.endswith("s"):
                return float(reset.rstrip("s"))
        except ValueError:
            pass
    return 5.0


def chat_json(prompt: str, system: str | None = None, timeout: int = 60) -> str:
    """Отправляет prompt Llama 3.3 70B, требует JSON-ответ (response_format).

    Возвращает сырую строку JSON — валидацию делает вызывающий код (см.
    triage.py), тем же способом, что и для локальной модели, чтобы сравнение
    было честным: обе модели проходят через один и тот же парсинг/retry.

    429 (TPM/RPM исчерпан) обрабатывается отдельно от прочих HTTP-ошибок:
    пауза по Retry-After/x-ratelimit-reset-* заголовку и повтор, а не
    немедленный crash всего сравнения на десятках уже посчитанных сообщений.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY не задан — добавь его в .env (см. .env.example)")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": GROQ_CHAT_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 400,
    }

    for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
        response = requests.post(
            GROQ_CHAT_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json=body, timeout=timeout,
        )
        if response.status_code == 429 and attempt < MAX_RETRIES_ON_RATE_LIMIT:
            wait_s = _retry_after_seconds(response)
            time.sleep(wait_s)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    raise RuntimeError(f"Groq: 429 после {MAX_RETRIES_ON_RATE_LIMIT} повторов")
