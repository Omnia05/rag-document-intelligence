from langchain_chroma import Chroma

from .embeddings import create_embeddings


def create_vectorstore(documents, persist_directory: str):
    """Create a Chroma vector store for the current session."""
    embeddings = create_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore

