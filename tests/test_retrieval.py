from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from retrieval import hybrid_search


def test_hybrid_search_returns_results():

    query = (
        "What is a Digital Health Platform?"
    )

    output = hybrid_search(
        query=query
    )

    assert "results" in output

    assert len(
        output["results"]
    ) > 0


def test_retrieval_result_has_required_fields():

    query = (
        "digital health platform"
    )

    output = hybrid_search(
        query=query
    )

    first_result = (
        output["results"][0]
    )

    assert "document" in first_result
    assert "metadata" in first_result
    assert "retrievers" in first_result
    assert "rrf_score" in first_result


def test_retrieved_document_is_not_empty():

    query = (
        "digital health infrastructure"
    )

    output = hybrid_search(
        query=query
    )

    first_result = (
        output["results"][0]
    )

    document = (
        first_result["document"]
    )

    assert isinstance(
        document,
        str
    )

    assert document.strip() != ""


def test_retrieval_metadata_contains_page():

    query = (
        "health information systems"
    )

    output = hybrid_search(
        query=query
    )

    first_result = (
        output["results"][0]
    )

    metadata = (
        first_result["metadata"]
    )

    assert "page" in metadata