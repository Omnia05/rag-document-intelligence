import tiktoken

from langchain_text_splitters import CharacterTextSplitter


def count_tokens(text: str) -> int:
    """Count tokens using the GPT-4o-mini tokenizer."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def filter_chunks_by_tokens(
    chunks,
    max_total_tokens: int = 250000,
    max_chunk_tokens: int = 8000
):
    """Filter chunks based on individual and total token limits."""
    filtered_chunks = []
    total_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk.page_content)

        if chunk_tokens > max_chunk_tokens:
            continue

        if total_tokens + chunk_tokens > max_total_tokens:
            break

        filtered_chunks.append(chunk)
        total_tokens += chunk_tokens

    return filtered_chunks


def split_documents(
    documents,
    chunk_size: int,
    chunk_overlap: int
):
    """Split documents into smaller chunks."""
    splitter = CharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap)
    )

    chunks = splitter.split_documents(documents)

    return filter_chunks_by_tokens(chunks)