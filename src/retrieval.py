from pathlib import Path
import time
import re
import chromadb


from bm25 import load_bm25_index, search_bm25


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "digital_health_handbook"

MODEL_NAME = "all-MiniLM-L6-v2"

VECTOR_TOP_K = 5
BM25_TOP_K = 5
FINAL_TOP_K = 5

RRF_K = 60


# ============================================================
# LAZY-LOADED RESOURCES
# ============================================================

client = None
collection = None
model = None
bm25_index_data = None


def get_collection():

    global client, collection

    if collection is None:

        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    return collection


def get_embedding_model():

    global model

    if model is None:

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            MODEL_NAME
        )

    return model


def get_bm25_index():

    global bm25_index_data

    if bm25_index_data is None:

        bm25_index_data = load_bm25_index()

    return bm25_index_data


# ============================================================
# VECTOR SEARCH
# ============================================================

def search_vector(
    query,
    top_k=VECTOR_TOP_K
):
    """
    Search ChromaDB using semantic vector similarity.
    """

    current_model = get_embedding_model()
    current_collection = get_collection()

    # --------------------------------------------------------
    # Convert query into embedding
    # --------------------------------------------------------

    start = time.perf_counter()

    query_embedding = current_model.encode(
        query
    )

    embedding_time = (
        time.perf_counter() - start
    )


    # --------------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------------

    start = time.perf_counter()

    results = current_collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieval_time = (
        time.perf_counter() - start
    )


    # --------------------------------------------------------
    # Convert Chroma output into common format
    # --------------------------------------------------------

    vector_results = []

    for i in range(
        len(results["ids"][0])
    ):

        vector_results.append({

            "id":
                results["ids"][0][i],

            "document":
                results["documents"][0][i],

            "metadata":
                results["metadatas"][0][i],

            "distance":
                float(
                    results["distances"][0][i]
                ),

            "retriever":
                "vector"
        })


    return {
        "results": vector_results,
        "embedding_time": embedding_time,
        "retrieval_time": retrieval_time
    }


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    final_top_k=FINAL_TOP_K,
    rrf_k=RRF_K
):
    """
    Combine vector-search and BM25 results using
    Reciprocal Rank Fusion (RRF).

    RRF score:

        1 / (rrf_k + rank)

    Documents appearing in both retrieval lists
    receive contributions from both.
    """

    fused_scores = {}

    result_map = {}

    source_map = {}


    # --------------------------------------------------------
    # Add vector results
    # --------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1
    ):

        doc_id = result["id"]

        score = (
            1 / (rrf_k + rank)
        )

        fused_scores[doc_id] = (
            fused_scores.get(
                doc_id,
                0
            )
            + score
        )

        result_map[doc_id] = result.copy()

        source_map.setdefault(
            doc_id,
            []
        ).append("vector")


    # --------------------------------------------------------
    # Add BM25 results
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        doc_id = result["id"]

        score = (
            1 / (rrf_k + rank)
        )

        fused_scores[doc_id] = (
            fused_scores.get(
                doc_id,
                0
            )
            + score
        )

        if doc_id not in result_map:

            result_map[doc_id] = (
                result.copy()
            )

        source_map.setdefault(
            doc_id,
            []
        ).append("bm25")


    # --------------------------------------------------------
    # Rank by final RRF score
    # --------------------------------------------------------

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )[:final_top_k]


    final_results = []

    for doc_id in ranked_ids:

        result = (
            result_map[doc_id].copy()
        )

        result["rrf_score"] = (
            fused_scores[doc_id]
        )

        result["retrievers"] = (
            source_map[doc_id]
        )

        final_results.append(
            result
        )


    return final_results

# ===========================================================
# DECOMPOSING QUERY
# ===========================================================
def decompose_query(query: str) -> list[str]:
    parts = re.split(
        r"\s+and\s+",
        query,
        flags=re.IGNORECASE
    )

    sub_queries = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    return sub_queries
# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    vector_top_k=VECTOR_TOP_K,
    bm25_top_k=BM25_TOP_K,
    final_top_k=FINAL_TOP_K
):
    """
    Perform hybrid retrieval with query decomposition:

    1. Decompose compound query
    2. Vector search for each sub-query
    3. BM25 search for each sub-query
    4. Reciprocal Rank Fusion for each sub-query
    5. Merge and deduplicate results
    6. Return final Top-K chunks
    """

    hybrid_start = time.perf_counter()

    sub_queries = decompose_query(query)

    print(f"Sub-queries: {sub_queries}")

    all_results = []

    total_embedding_time = 0
    total_vector_time = 0
    total_bm25_time = 0
    total_fusion_time = 0

    all_vector_results = []
    all_bm25_results = []

    

    # --------------------------------------------------------
    # Search each sub-query separately
    # --------------------------------------------------------

    for sub_query in sub_queries:

        print(f"\nSearching sub-query: {sub_query}")

        # ----------------------------------------------------
        # Vector search
        # ----------------------------------------------------

        vector_output = search_vector(
            query=sub_query,
            top_k=vector_top_k
        )

        vector_results = vector_output["results"]

        total_embedding_time += (
            vector_output["embedding_time"]
        )

        total_vector_time += (
            vector_output["retrieval_time"]
        )

        all_vector_results.extend(
            vector_results
        )

        # ----------------------------------------------------
        # BM25 search
        # ----------------------------------------------------
       
        bm25_start = time.perf_counter()

        current_bm25_index = get_bm25_index()

        bm25_results = search_bm25(
            query=sub_query,
            index_data=current_bm25_index,
            top_k=bm25_top_k
        )

        bm25_time = (
            time.perf_counter()
            - bm25_start
        )

        total_bm25_time += bm25_time

        all_bm25_results.extend(
            bm25_results
        )

        # ----------------------------------------------------
        # Fuse this sub-query
        # ----------------------------------------------------

        fusion_start = time.perf_counter()

        fused_results = reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            final_top_k=final_top_k
        )

        fusion_time = (
            time.perf_counter()
            - fusion_start
        )

        total_fusion_time += fusion_time

        all_results.extend(
            fused_results
        )

    
    # --------------------------------------------------------
    # Deduplicate merged results
    # --------------------------------------------------------

    unique_results = {}

    for result in all_results:

        result_id = result["id"]

        if result_id not in unique_results:

            unique_results[result_id] = result.copy()

        else:

            unique_results[result_id]["rrf_score"] += (
                result["rrf_score"]
            )

            existing_retrievers = set(
                unique_results[result_id].get(
                    "retrievers",
                    []
                )
            )

            new_retrievers = set(
                result.get(
                    "retrievers",
                    []
                )
            )

            unique_results[result_id]["retrievers"] = list(
                existing_retrievers | new_retrievers
            )

    # --------------------------------------------------------
    # Sort merged results
    # --------------------------------------------------------

    final_results = sorted(
        unique_results.values(),
        key=lambda result: result["rrf_score"],
        reverse=True
    )[:final_top_k]

    # --------------------------------------------------------
    # Total timing
    # --------------------------------------------------------

    total_hybrid_time = (
        time.perf_counter()
        - hybrid_start
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
    "results": final_results,

    "vector_results": all_vector_results,
    "bm25_results": all_bm25_results,

    "timing": {
    "embedding_time": total_embedding_time,
    "vector_retrieval_time": total_vector_time,
    "bm25_time": total_bm25_time,
    "fusion_time": total_fusion_time,
    "total_time": total_hybrid_time,
}
}