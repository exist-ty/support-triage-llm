import json

import pytest
from pydantic import ValidationError

from src import triage


def test_parse_llm_response_valid_json():
    raw = json.dumps({
        "category": "delivery_status",
        "sentiment": "negative",
        "priority": "high",
        "confidence": 0.9,
        "suggested_reply": "Проверим трек-номер и вернёмся с ответом.",
    })
    result = triage.parse_llm_response(raw)
    assert result.category == "delivery_status"
    assert result.priority == "high"


def test_parse_llm_response_strips_markdown_fences():
    raw = "```json\n" + json.dumps({
        "category": "return",
        "sentiment": "neutral",
        "priority": "medium",
        "confidence": 0.7,
        "suggested_reply": "Оформим возврат по инструкции.",
    }) + "\n```"
    result = triage.parse_llm_response(raw)
    assert result.category == "return"


def test_unknown_category_coerced_to_other():
    raw = json.dumps({
        "category": "something_the_model_invented",
        "sentiment": "neutral",
        "priority": "medium",
        "confidence": 0.5,
        "suggested_reply": "...",
    })
    result = triage.parse_llm_response(raw)
    assert result.category == "other"


def test_unknown_sentiment_and_priority_coerced_to_defaults():
    raw = json.dumps({
        "category": "praise",
        "sentiment": "very happy",
        "priority": "urgent!!!",
        "confidence": 0.8,
        "suggested_reply": "...",
    })
    result = triage.parse_llm_response(raw)
    assert result.sentiment == "neutral"
    assert result.priority == "medium"


def test_confidence_out_of_range_raises():
    raw = json.dumps({
        "category": "praise",
        "sentiment": "positive",
        "priority": "low",
        "confidence": 1.5,
        "suggested_reply": "...",
    })
    with pytest.raises(ValidationError):
        triage.parse_llm_response(raw)


def test_classify_message_retries_on_invalid_json_then_succeeds(monkeypatch):
    valid_response = json.dumps({
        "category": "defect",
        "sentiment": "negative",
        "priority": "high",
        "confidence": 0.85,
        "suggested_reply": "Оформим обмен на новый товар.",
    })
    responses = iter(["not a json at all", valid_response])
    monkeypatch.setattr(triage, "chat_json", lambda prompt, system=None: next(responses))

    result = triage.classify_message("Товар не работает", retrieved_docs=[])
    assert result.category == "defect"


def test_classify_message_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(triage, "chat_json", lambda prompt, system=None: "still not json")

    with pytest.raises(RuntimeError):
        triage.classify_message("Товар не работает", retrieved_docs=[], max_retries=1)
