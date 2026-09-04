from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def create_chunks(pages: list[dict]) -> list[dict]:
    """Split extracted pages into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                }
            )

    return chunks