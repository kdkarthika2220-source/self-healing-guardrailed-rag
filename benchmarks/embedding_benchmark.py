
from pathlib import Path
import sys
import time

from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion import extract_pages_from_pdf
from chunking import create_chunks


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "digital_health_handbook.pdf"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZES = [8, 16, 32, 64]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EMBEDDING BATCH-SIZE BENCHMARK")
print("=" * 70)

print("\nLoading PDF...")

pages = extract_pages_from_pdf(PDF_PATH)

chunks = create_chunks(pages)

texts = [chunk["text"] for chunk in chunks]

print("Pages:", len(pages))
print("Chunks:", len(texts))


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("\nLoading model...")

model_start = time.perf_counter()

model = SentenceTransformer(MODEL_NAME)

model_load_time = time.perf_counter() - model_start

print(f"Model load time: {model_load_time:.2f} seconds")

print("Embedding dimension:", model.get_embedding_dimension())


# ============================================================
# BENCHMARK DIFFERENT BATCH SIZES
# ============================================================

results = []


for batch_size in BATCH_SIZES:

    print("\n" + "=" * 70)
    print(f"TESTING BATCH SIZE: {batch_size}")
    print("=" * 70)

    start_time = time.perf_counter()

    try:

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        elapsed = time.perf_counter() - start_time

        throughput = len(texts) / elapsed

        memory_mb = embeddings.nbytes / (1024 * 1024)

        results.append(
            {
                "batch_size": batch_size,
                "time": elapsed,
                "throughput": throughput,
                "memory_mb": memory_mb,
                "status": "SUCCESS",
            }
        )

        print("\nResult:")
        print(f"Batch size:        {batch_size}")
        print(f"Embedding time:    {elapsed:.4f} seconds")
        print(f"Throughput:        {throughput:.2f} chunks/sec")
        print(f"Vector memory:     {memory_mb:.2f} MB")
        print("Status:            SUCCESS")

    except Exception as e:

        results.append(
            {
                "batch_size": batch_size,
                "time": None,
                "throughput": None,
                "memory_mb": None,
                "status": f"FAILED: {e}",
            }
        )

        print("\nFAILED")
        print("Error:", e)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BATCH-SIZE BENCHMARK SUMMARY")
print("=" * 70)

print(
    f"{'Batch':<10}"
    f"{'Time (sec)':<15}"
    f"{'Chunks/sec':<15}"
    f"{'Vector MB':<15}"
    f"{'Status'}"
)

print("-" * 70)

for result in results:

    if result["status"] == "SUCCESS":

        print(
            f"{result['batch_size']:<10}"
            f"{result['time']:<15.4f}"
            f"{result['throughput']:<15.2f}"
            f"{result['memory_mb']:<15.2f}"
            f"{result['status']}"
        )

    else:

        print(
            f"{result['batch_size']:<10}"
            f"{'-':<15}"
            f"{'-':<15}"
            f"{'-':<15}"
            f"{result['status']}"
        )


# ============================================================
# FIND BEST BATCH SIZE
# ============================================================

successful_results = [
    result
    for result in results
    if result["status"] == "SUCCESS"
]


if successful_results:

    best = max(
        successful_results,
        key=lambda result: result["throughput"]
    )

    print("\n" + "=" * 70)
    print("BEST BATCH SIZE")
    print("=" * 70)

    print("Batch size:", best["batch_size"])
    print(f"Throughput: {best['throughput']:.2f} chunks/sec")
    print(f"Time: {best['time']:.4f} seconds")

print("\nBenchmark complete.")
