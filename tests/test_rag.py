import pytest

from src.rag import reciprocal_rank_fusion, to_vector_literal


def test_to_vector_literal_formats_as_pgvector_array_string():
    assert to_vector_literal([0.1, -0.2, 3.0]) == "[0.1,-0.2,3.0]"


def test_to_vector_literal_handles_empty_list():
    assert to_vector_literal([]) == "[]"


def test_rrf_ranks_doc_agreed_on_by_both_lists_first():
    dense = [{"id": 1}, {"id": 2}, {"id": 3}]
    sparse = [{"id": 2}, {"id": 3}, {"id": 1}]

    fused = reciprocal_rank_fusion([dense, sparse])

    # id=2: rank 2 в dense + rank 1 в sparse -> лучший суммарный ранг
    assert fused[0]["id"] == 2


def test_rrf_includes_doc_found_by_only_one_list():
    dense = [{"id": 1}, {"id": 2}]
    sparse = [{"id": 3}]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert {doc["id"] for doc in fused} == {1, 2, 3}


def test_rrf_score_matches_sum_of_reciprocal_ranks():
    dense = [{"id": 1}, {"id": 2}]
    sparse = [{"id": 1}]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    doc_1 = next(doc for doc in fused if doc["id"] == 1)

    # id=1: rank 1 в обоих списках -> 1/(60+1) + 1/(60+1)
    assert doc_1["rrf_score"] == pytest.approx(2 / 61)


def test_rrf_preserves_doc_fields():
    dense = [{"id": 1, "title": "t", "content": "c"}]

    fused = reciprocal_rank_fusion([dense])

    assert fused[0]["title"] == "t"
    assert fused[0]["content"] == "c"
