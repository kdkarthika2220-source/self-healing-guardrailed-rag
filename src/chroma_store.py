from pathlib import Path
import time

import chromadb
from sentence_transformers import SentenceTransformer

from ingestion import extract_pages_from_pdf
from chunking import create_chunks


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "digital_health_handbook.pdf"
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

MODEL_NAME = "all-MiniLM-L6-v2"

# Selected from our batch-size benchmark
BATCH_SIZE = 16


# ============================================================
# MAIN INGESTION FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("CHROMA DB INGESTION")
    print("=" * 60)

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. VALIDATE PDF
    # --------------------------------------------------------

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # --------------------------------------------------------
    # 2. EXTRACT PDF
    # --------------------------------------------------------

    print("\n[1/4] Extracting PDF...")

    start = time.perf_counter()

    pages = extract_pages_from_pdf(
        PDF_PATH
    )

    extraction_time = (
        time.perf_counter() - start
    )

    print(
        f"Pages extracted : {len(pages)}"
    )
    print(
        f"Extraction time : "
        f"{extraction_time:.4f} sec"
    )

    # --------------------------------------------------------
    # 3. CREATE CHUNKS
    # --------------------------------------------------------

    print("\n[2/4] Creating chunks...")

    start = time.perf_counter()

    chunks = create_chunks(
        pages
    )

    chunking_time = (
        time.perf_counter() - start
    )

    if not chunks:
        raise ValueError(
            "No chunks were created from the PDF."
        )

    print(
        f"Chunks created  : {len(chunks)}"
    )
    print(
        f"Chunking time   : "
        f"{chunking_time:.4f} sec"
    )

    # --------------------------------------------------------
    # 4. GENERATE EMBEDDINGS
    # --------------------------------------------------------

    print(
        "\n[3/4] Generating embeddings..."
    )

    start = time.perf_counter()

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embedding_time = (
        time.perf_counter() - start
    )

    print(
        f"\nEmbedding shape : "
        f"{embeddings.shape}"
    )
    print(
        f"Embedding time  : "
        f"{embedding_time:.4f} sec"
    )
    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # 5. INITIALIZE CHROMADB
    # --------------------------------------------------------

    print(
        "\n[4/4] Storing vectors "
        "in ChromaDB..."
    )

    start = time.perf_counter()

    CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    # --------------------------------------------------------
    # 6. REBUILD COLLECTION
    # --------------------------------------------------------
    # We rebuild the collection completely so stale chunks
    # from previous ingestion runs cannot remain.

    existing_collections = [
        collection.name
        for collection
        in client.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:

        print(
            "Existing collection found. "
            "Rebuilding..."
        )

        client.delete_collection(
            name=COLLECTION_NAME
        )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    # --------------------------------------------------------
    # 7. PREPARE IDS AND METADATA
    # --------------------------------------------------------

    ids = [
        f"chunk_{i:04d}"
        for i in range(len(chunks))
    ]

    metadatas = []

    for i, chunk in enumerate(chunks):

        metadatas.append(
            {
                "source": chunk["source"],
                "page": int(
                    chunk["page"]
                ),
                "chunk_index": i,
            }
        )

    # --------------------------------------------------------
    # 8. STORE VECTORS
    # --------------------------------------------------------

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    storage_time = (
        time.perf_counter() - start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # 9. RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(
        f"Collection name : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Stored vectors  : "
        f"{collection.count()}"
    )

    print(
        f"Vector dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Storage path    : "
        f"{CHROMA_PATH}"
    )

    print("\nTiming:")

    print(
        f"Extraction      : "
        f"{extraction_time:.4f} sec"
    )

    print(
        f"Chunking        : "
        f"{chunking_time:.4f} sec"
    )

    print(
        f"Embedding       : "
        f"{embedding_time:.4f} sec"
    )

    print(
        f"Chroma storage  : "
        f"{storage_time:.4f} sec"
    )

    print(
        f"Total           : "
        f"{total_time:.4f} sec"
    )

    # --------------------------------------------------------
    # 10. SAMPLE METADATA
    # --------------------------------------------------------

    sample = collection.get(
        ids=["chunk_0000"],
        include=["metadatas"]
    )

    if sample["metadatas"]:

        print("\nSample metadata:")
        print(
            sample["metadatas"][0]
        )

    print(
        "\nChromaDB is ready!"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()