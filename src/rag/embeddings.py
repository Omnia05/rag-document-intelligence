from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL


def create_embeddings():
    """Create the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

