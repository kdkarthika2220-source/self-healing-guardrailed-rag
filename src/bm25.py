from pathlib import Path
import pickle

import chromadb
from rank_bm25 import BM25Okapi


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BM25_INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "bm25_index.pkl"
)

CHROMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "chroma"
)


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "digital_health_handbook"


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase whitespace-separated tokens.
    """

    return text.lower().split()


# ============================================================
# BUILD BM25 INDEX
# ============================================================

def build_bm25_index(
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
):
    """
    Build and persist a BM25 index from documents.
    """

    if not documents:
        raise ValueError(
            "Cannot build BM25 index: no documents provided."
        )

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    index_data = {
        "bm25": bm25,
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }

    BM25_INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        BM25_INDEX_PATH,
        "wb"
    ) as file:

        pickle.dump(
            index_data,
            file
        )

    print(
        f"BM25 index saved to: "
        f"{BM25_INDEX_PATH}"
    )


# ============================================================
# LOAD BM25 INDEX
# ============================================================

def load_bm25_index() -> dict:
    """
    Load the persisted BM25 index from disk.
    """

    if not BM25_INDEX_PATH.exists():

        raise FileNotFoundError(
            "BM25 index not found at: "
            f"{BM25_INDEX_PATH}"
        )

    with open(
        BM25_INDEX_PATH,
        "rb"
    ) as file:

        index_data = pickle.load(file)

    return index_data


# ============================================================
# BM25 SEARCH
# ============================================================

def search_bm25(
    query: str,
    index_data: dict,
    top_k: int = 3,
) -> list[dict]:
    """
    Search the BM25 index and return
    the highest-scoring keyword matches.
    """

    if not query.strip():
        return []

    bm25 = index_data["bm25"]
    documents = index_data["documents"]
    metadatas = index_data["metadatas"]
    ids = index_data["ids"]

    tokenized_query = tokenize(
        query
    )

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for index in ranked_indices:

        score = float(
            scores[index]
        )

        # Do not inject unrelated zero-score
        # documents into hybrid retrieval.
        if score <= 0:
            continue

        results.append(
            {
                "id": ids[index],
                "document": documents[index],
                "metadata": metadatas[index],
                "score": score,
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# BUILD INDEX FROM CHROMADB
# ============================================================

def rebuild_bm25_from_chroma():
    """
    Rebuild the BM25 index using the documents
    currently stored in ChromaDB.
    """

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    all_data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    ids = all_data["ids"]

    print("=" * 60)
    print("BM25 INDEX BUILD")
    print("=" * 60)

    print(
        f"\nDocuments loaded from Chroma: "
        f"{len(documents)}"
    )

    build_bm25_index(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    print(
        f"Indexed documents: "
        f"{len(documents)}"
    )

    print(
        "\nBM25 index is ready!"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    rebuild_bm25_from_chroma()