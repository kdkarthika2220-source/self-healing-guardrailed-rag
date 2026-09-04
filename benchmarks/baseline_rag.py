from pathlib import Path
import os
import time

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "digital_health_handbook"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 3


# ---------------------------------------------------------
# Check API key
# ---------------------------------------------------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not configured."
    )


# ---------------------------------------------------------
# Initialize clients
# ---------------------------------------------------------

print("=" * 60)
print("BASELINE RAG - GENERATION")
print("=" * 60)

print("\nLoading ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(f"Documents: {collection.count()}")


print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


print("\nConnecting to Groq...")

groq_client = Groq(
    api_key=api_key
)

print("Groq client ready.")


# ---------------------------------------------------------
# Question
# ---------------------------------------------------------

query = "What is the capital of France?"


print("\n" + "=" * 60)
print("USER QUESTION")
print("=" * 60)

print(query)


# ---------------------------------------------------------
# Step 1: Embed question
# ---------------------------------------------------------

start = time.perf_counter()

query_embedding = embedding_model.encode(query)

embedding_time = time.perf_counter() - start


# ---------------------------------------------------------
# Step 2: Retrieve relevant chunks
# ---------------------------------------------------------

start = time.perf_counter()

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=TOP_K,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)

retrieval_time = time.perf_counter() - start


documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


# ---------------------------------------------------------
# Build context
# ---------------------------------------------------------

context_parts = []

for i, (document, metadata) in enumerate(
    zip(documents, metadatas),
    start=1
):

    context_parts.append(
        f"""
SOURCE {i}
Page: {metadata['page']}
Content:
{document}
"""
    )


context = "\n".join(context_parts)


# ---------------------------------------------------------
# Step 3: Generate answer
# ---------------------------------------------------------

system_prompt = """
You are a question-answering assistant.

Answer the user's question ONLY using the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the context does not contain enough information to answer,
   say exactly:
   "I don't have enough information in the provided document."
4. Keep the answer concise and factual.
5. Mention the relevant page number when possible.
"""


user_prompt = f"""
Context:

{context}

Question:
{query}
"""


print("\n" + "=" * 60)
print("GENERATING ANSWER")
print("=" * 60)

start = time.perf_counter()

response = groq_client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    temperature=0,
    reasoning_effort="low"
)

generation_time = time.perf_counter() - start


# ---------------------------------------------------------
# Final answer
# ---------------------------------------------------------

answer = response.choices[0].message.content


print("\n" + "=" * 60)
print("RAG ANSWER")
print("=" * 60)

print(answer)


# ---------------------------------------------------------
# Performance
# ---------------------------------------------------------

total_time = (
    embedding_time
    + retrieval_time
    + generation_time
)

print("\n" + "=" * 60)
print("PERFORMANCE")
print("=" * 60)

print(f"Question embedding : {embedding_time:.4f} sec")
print(f"Chroma retrieval   : {retrieval_time:.4f} sec")
print(f"LLM generation     : {generation_time:.4f} sec")
print(f"Total RAG latency  : {total_time:.4f} sec")

print("\nBaseline RAG test complete! ")