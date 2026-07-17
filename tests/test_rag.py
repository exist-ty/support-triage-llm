from src.rag import to_vector_literal


def test_to_vector_literal_formats_as_pgvector_array_string():
    assert to_vector_literal([0.1, -0.2, 3.0]) == "[0.1,-0.2,3.0]"


def test_to_vector_literal_handles_empty_list():
    assert to_vector_literal([]) == "[]"
