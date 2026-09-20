import os

from .ingestion import load_documents_with_size_limit
from .preprocessing import split_documents
from .vectorstore import create_vectorstore
from .chat import create_conversation_chain


def process_documents(
    knowledge_base_dir: str,
    max_file_size_mb: float,
    chunk_size: int,
    chunk_overlap: int,
    session_dir: str,
    num_chunks: int = 25
):
    """Run the complete document processing pipeline."""

    max_size_bytes = int(max_file_size_mb * 1024 * 1024)

    all_documents = []
    skipped_files = []
    processing_log = []

    processing_log.append(
        f"Knowledge Base Directory: {knowledge_base_dir}"
    )
    processing_log.append(
        f"Maximum File Size: {max_file_size_mb} MB"
    )
    processing_log.append(
        f"Chunk Size: {chunk_size} characters"
    )
    processing_log.append(
        f"Chunk Overlap: {chunk_overlap} characters"
    )
    processing_log.append(
        f"Retrieval K: {num_chunks}"
    )
    processing_log.append("")
    processing_log.append("=== Document Ingestion ===")

    directories = [knowledge_base_dir]

    directories.extend(
        os.path.join(knowledge_base_dir, item)
        for item in os.listdir(knowledge_base_dir)
        if os.path.isdir(
            os.path.join(knowledge_base_dir, item)
        )
    )

    for directory in directories:
        doc_type = os.path.basename(directory)

        processing_log.append(
            f"\nProcessing folder: {doc_type}"
        )

        documents, skipped = load_documents_with_size_limit(
            directory,
            doc_type,
            max_size_bytes,
            recursive=(directory != knowledge_base_dir)
        )

        processing_log.append(
            f"  Documents loaded: {len(documents)}"
        )

        file_counts = {}

        for doc in documents:
            file_name = doc.metadata.get(
                "file_name",
                "Unknown file"
            )

            file_counts[file_name] = (
                file_counts.get(file_name, 0) + 1
            )

        for file_name, count in file_counts.items():
            processing_log.append(
                f"    - {file_name}: {count} document(s)"
            )

        all_documents.extend(documents)
        skipped_files.extend(skipped)

        processing_log.append(
            f"  Pages loaded: {len(documents)}"
        )

        if skipped:
            processing_log.append(
                f"  Files skipped: {len(skipped)}"
            )

            for path, reason in skipped:
                processing_log.append(
                    f"    - {path}: {reason}"
                )

    if not all_documents:
        raise ValueError(
            "No valid PDF documents were found."
        )

    processing_log.append("")
    processing_log.append("=== Chunking ===")

    processing_log.append(
        f"Documents before chunking: "
        f"{len(all_documents)}"
    )

    chunks = split_documents(
        all_documents,
        chunk_size,
        chunk_overlap
    )

    if not chunks:
        raise ValueError(
            "No valid chunks were created."
        )

    processing_log.append(
        f"Chunks created: {len(chunks)}"
    )

    processing_log.append("")
    processing_log.append("=== Vector Store ===")

    os.makedirs(session_dir, exist_ok=True)

    vectorstore = create_vectorstore(
        chunks,
        session_dir
    )

    processing_log.append(
        f"Chroma vector store created at:"
    )
    processing_log.append(
        f"  {session_dir}"
    )

    processing_log.append("")
    processing_log.append("=== Conversational Retrieval ===")

    conversation_chain = create_conversation_chain(
        vectorstore,
        num_chunks
    )

    processing_log.append(
        f"Conversation chain created successfully"
    )
    processing_log.append(
        f"Retriever configured with k={num_chunks}"
    )

    processing_log.append("")
    processing_log.append("=== Processing Complete ===")

    processing_log.append(
        f"Total documents loaded: {len(all_documents)}"
    )
    processing_log.append(
        f"Total chunks: {len(chunks)}"
    )
    processing_log.append(
        f"Total files skipped: {len(skipped_files)}"
    )

    return {
        "vectorstore": vectorstore,
        "conversation_chain": conversation_chain,
        "documents": len(all_documents),
        "chunks": len(chunks),
        "skipped_files": skipped_files,
        "processing_log": processing_log
    }