"""Тонкий клиент поверх локального Ollama HTTP API. Никакого внешнего
SDK — только requests, т.к. Ollama предоставляет обычный REST API."""
from __future__ import annotations

import os

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")


def chat_json(prompt: str, system: str | None = None, timeout: int = 120) -> str:
    """Отправляет prompt модели генерации, требует JSON-ответ (format=json).

    Возвращает сырую строку JSON — валидацию делает вызывающий код
    (см. triage.py), т.к. модель на 3B параметрах иногда возвращает
    JSON с лишними полями или обёрткой.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "format": "json",
            "stream": False,
            # num_predict ограничивает суммарную длину ответа: на слабом CPU
            # (без CUDA — см. README) каждый лишний токен генерации стоит
            # заметного времени, а suggested_reply не должен быть длиннее
            # пары предложений.
            "options": {"temperature": 0.2, "num_predict": 150},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def embed(texts: list[str], timeout: int = 60) -> list[list[float]]:
    """Возвращает эмбеддинги для списка текстов через модель EMBED_MODEL."""
    response = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["embeddings"]
