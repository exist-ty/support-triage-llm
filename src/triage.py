"""Классификация обращения клиента через локальную LLM (Qwen2.5-3B-Instruct
поверх Ollama) с RAG-контекстом из kb_documents. Структурированный вывод
валидируется pydantic — модель на 3B параметров не всегда идеально следует
JSON-схеме, поэтому есть один retry и мягкий fallback вместо падения
пайплайна на одном "плохом" сообщении."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, field_validator

from src.ollama_client import CHAT_MODEL, chat_json

ALLOWED_CATEGORIES = {
    "delivery_status", "return", "defect", "payment", "cancel",
    "loyalty", "exchange", "wholesale", "complaint_rude", "praise", "other",
}
ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}
ALLOWED_PRIORITIES = {"low", "medium", "high"}

SYSTEM_PROMPT = (
    "Ты — ассистент поддержки интернет-магазина электроники. Классифицируешь "
    "обращения клиентов и предлагаешь ответ на основе выдержек из базы знаний. "
    "Отвечай ТОЛЬКО валидным JSON без пояснений."
)

PROMPT_TEMPLATE = """Обращение клиента:
"{message}"

Выдержки из базы знаний, которые могут быть релевантны:
{context}

Верни JSON строго с полями:
- category: одно из {categories}
- sentiment: одно из {sentiments}
- priority: одно из {priorities}
- confidence: число от 0 до 1 (уверенность в категории)
- suggested_reply: короткий черновик ответа клиенту на русском (1-3 предложения),
  используй базу знаний, если она релевантна, иначе ответь по существу без выдумывания фактов
"""


class TriageResult(BaseModel):
    category: str
    sentiment: str
    priority: str
    confidence: float = Field(ge=0, le=1)
    suggested_reply: str

    @field_validator("category")
    @classmethod
    def _coerce_category(cls, value: str) -> str:
        return value if value in ALLOWED_CATEGORIES else "other"

    @field_validator("sentiment")
    @classmethod
    def _coerce_sentiment(cls, value: str) -> str:
        return value if value in ALLOWED_SENTIMENTS else "neutral"

    @field_validator("priority")
    @classmethod
    def _coerce_priority(cls, value: str) -> str:
        return value if value in ALLOWED_PRIORITIES else "medium"


def _strip_code_fences(raw: str) -> str:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return match.group(0) if match else raw


def parse_llm_response(raw: str) -> TriageResult:
    cleaned = _strip_code_fences(raw)
    data = json.loads(cleaned)
    return TriageResult.model_validate(data)


def build_prompt(message_text: str, retrieved_docs: list[dict]) -> str:
    context = "\n".join(f"- {doc['title']}: {doc['content']}" for doc in retrieved_docs) or "(нет релевантных документов)"
    return PROMPT_TEMPLATE.format(
        message=message_text,
        context=context,
        categories=", ".join(sorted(ALLOWED_CATEGORIES)),
        sentiments=", ".join(sorted(ALLOWED_SENTIMENTS)),
        priorities=", ".join(sorted(ALLOWED_PRIORITIES)),
    )


def classify_message(message_text: str, retrieved_docs: list[dict], max_retries: int = 2) -> TriageResult:
    prompt = build_prompt(message_text, retrieved_docs)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        raw = chat_json(prompt, system=SYSTEM_PROMPT)
        try:
            return parse_llm_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            prompt = prompt + "\n\nВАЖНО: предыдущий ответ был невалидным JSON. Верни только JSON-объект."

    raise RuntimeError(f"LLM ({CHAT_MODEL}) не вернула валидный JSON за {max_retries + 1} попыток: {last_error}")
