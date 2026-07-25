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

import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def chat_json(prompt: str, system: str | None = None, timeout: int = 60) -> str:
    """Отправляет prompt Llama 3.3 70B, требует JSON-ответ (response_format).

    Возвращает сырую строку JSON — валидацию делает вызывающий код (см.
    triage.py), тем же способом, что и для локальной модели, чтобы сравнение
    было честным: обе модели проходят через один и тот же парсинг/retry.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY не задан — добавь его в .env (см. .env.example)")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_CHAT_MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 400,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
