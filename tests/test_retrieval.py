from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from retrieval import reciprocal_rank_fusion, decompose_query


def test_query_decomposition():

    query = (
        "digital health platform and health information systems"
    )

    parts = decompose_query(query)

    assert parts == [
        "digital health platform",
        "health information systems"
    ]


def test_rrf_returns_results():

    vector_results = [
        {
            "id": "doc1",
            "document": "Digital health platform overview",
            "metadata": {"page": 1},
            "distance": 0.1,
            "retriever": "vector"
        }
    ]

    bm25_results = [
        {
            "id": "doc2",
            "document": "Health information infrastructure",
            "metadata": {"page": 2},
            "score": 5.0,
            "retriever": "bm25"
        }
    ]

    results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results
    )

    assert len(results) > 0


def test_rrf_result_has_required_fields():

    vector_results = [
        {
            "id": "doc1",
            "document": "Digital health platform",
            "metadata": {"page": 10},
            "distance": 0.2,
            "retriever": "vector"
        }
    ]

    bm25_results = []

    results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results
    )

    first_result = results[0]

    assert "document" in first_result
    assert "metadata" in first_result
    assert "retrievers" in first_result
    assert "rrf_score" in first_result


def test_rrf_combines_duplicate_document_scores():

    vector_results = [
        {
            "id": "doc1",
            "document": "Digital health platform",
            "metadata": {"page": 5},
            "distance": 0.1,
            "retriever": "vector"
        }
    ]

    bm25_results = [
        {
            "id": "doc1",
            "document": "Digital health platform",
            "metadata": {"page": 5},
            "score": 8.0,
            "retriever": "bm25"
        }
    ]

    results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results
    )

    first_result = results[0]

    assert set(
        first_result["retrievers"]
    ) == {
        "vector",
        "bm25"
    }

    assert first_result["rrf_score"] > (
        1 / 61
    )