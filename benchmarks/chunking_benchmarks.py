

from pathlib import Path
import sys
import time
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion import extract_pages_from_pdf


PDF_PATH = PROJECT_ROOT / "data" / "documents" / "digital_health_handbook.pdf"


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


# ============================================================
# CHUNKING FUNCTION
# ============================================================

def create_chunks(pages: list[dict]) -> list[dict]:
    """Split extracted pages into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append(
                {
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                }
            )

    return chunks


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RAG CHUNKING BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. PDF EXTRACTION + CLEANING
    # --------------------------------------------------------

    extraction_start = time.perf_counter()

    pages = extract_pages_from_pdf(PDF_PATH)

    extraction_time = time.perf_counter() - extraction_start

    print("\nPDF EXTRACTION")
    print("-" * 70)
    print("Total pages:", len(pages))
    print(f"Extraction + cleaning time: {extraction_time:.4f} seconds")

    # --------------------------------------------------------
    # 2. CHUNKING
    # --------------------------------------------------------

    chunking_start = time.perf_counter()

    chunks = create_chunks(pages)

    chunking_time = time.perf_counter() - chunking_start

    print("\nCHUNKING")
    print("-" * 70)
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Total chunks:", len(chunks))
    print(f"Actual chunking time: {chunking_time:.4f} seconds")

    # --------------------------------------------------------
    # 3. CHUNK SIZE STATISTICS
    # --------------------------------------------------------

    chunk_lengths = [len(chunk["text"]) for chunk in chunks]

    average_length = sum(chunk_lengths) / len(chunk_lengths)

    print("\nCHUNK SIZE STATISTICS")
    print("-" * 70)
    print(f"Average characters: {average_length:.2f}")
    print("Minimum characters:", min(chunk_lengths))
    print("Maximum characters:", max(chunk_lengths))

    # --------------------------------------------------------
    # 4. CHUNK DISTRIBUTION
    # --------------------------------------------------------

    distribution = Counter()

    for length in chunk_lengths:

        if length < 200:
            distribution["< 200"] += 1

        elif length < 500:
            distribution["200 - 499"] += 1

        elif length < 1000:
            distribution["500 - 999"] += 1

        elif length < 1500:
            distribution["1000 - 1499"] += 1

        elif length < 2000:
            distribution["1500 - 1999"] += 1

        else:
            distribution["2000"] += 1

    print("\nCHUNK DISTRIBUTION")
    print("-" * 70)

    for category, count in distribution.items():

        percentage = (count / len(chunks)) * 100

        print(
            f"{category:>10} characters : "
            f"{count:>4} chunks "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 5. VERY SMALL CHUNKS
    # --------------------------------------------------------

    small_chunks = [
        chunk
        for chunk in chunks
        if len(chunk["text"]) < 200
    ]

    print("\nVERY SMALL CHUNKS")
    print("-" * 70)
    print("Chunks below 200 characters:", len(small_chunks))

    # --------------------------------------------------------
    # 6. SHOW EXAMPLES OF SMALL CHUNKS
    # --------------------------------------------------------

    if small_chunks:

        print("\nExamples of very small chunks:")

        for i, chunk in enumerate(small_chunks[:5], start=1):

            print("\n" + "-" * 50)
            print(f"SMALL CHUNK {i}")
            print("-" * 50)

            print("Page:", chunk["page"])
            print("Characters:", len(chunk["text"]))
            print("Text:")
            print(chunk["text"])

    # --------------------------------------------------------
    # 7. FIRST NORMAL CHUNK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST CHUNK")
    print("=" * 70)

    first_chunk = chunks[0]

    print("Source:", first_chunk["source"])
    print("Page:", first_chunk["page"])
    print("Characters:", len(first_chunk["text"]))
    print("\nText:")
    print(first_chunk["text"])

    # --------------------------------------------------------
    # 8. TOTAL PROCESSING TIME
    # --------------------------------------------------------

    total_time = extraction_time + chunking_time

    print("\n" + "=" * 70)
    print("TOTAL PROCESSING TIME")
    print("=" * 70)

    print(f"Extraction + cleaning: {extraction_time:.4f} seconds")
    print(f"Chunking:              {chunking_time:.4f} seconds")
    print(f"Total:                 {total_time:.4f} seconds")

    print("\nBenchmark complete.")
