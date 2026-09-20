import glob
import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx"
}


def check_file_size(file_path: str, max_size_bytes: int):
    """Check whether a file is within the allowed size limit."""
    try:
        file_size = os.path.getsize(file_path)
        return file_size <= max_size_bytes, file_size
    except OSError:
        return False, 0


def add_metadata(doc, doc_type: str, file_path: str):
    """Add document type and file information to a document."""
    doc.metadata["doc_type"] = doc_type
    doc.metadata["file_path"] = file_path
    doc.metadata["file_name"] = os.path.basename(file_path)
    doc.metadata["file_extension"] = os.path.splitext(
        file_path
    )[1].lower()

    return doc


def get_loader(file_path: str):
    """Return the appropriate loader for a file type."""
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return PyPDFLoader(file_path)

    if extension in {".txt", ".md"}:
        return TextLoader(
            file_path,
            encoding="utf-8"
        )

    if extension == ".docx":
        return Docx2txtLoader(file_path)

    return None


def load_documents_with_size_limit(
    folder_path: str,
    doc_type: str,
    max_size_bytes: int,
    recursive: bool = True
):
    """Load supported documents recursively with size restrictions."""

    if recursive:
        files = glob.glob(
            os.path.join(folder_path, "**/*"),
            recursive=True
        )
    else:
        files = glob.glob(
            os.path.join(folder_path, "*")
        )

    loaded_docs = []
    skipped_files = []

    for file_path in files:

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(
            file_path
        )[1].lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        is_valid_size, file_size = check_file_size(
            file_path,
            max_size_bytes
        )

        if not is_valid_size:
            file_size_mb = file_size / 1024 / 1024

            skipped_files.append(
                (
                    file_path,
                    f"File too large: {file_size_mb:.2f} MB"
                )
            )

            continue

        try:
            loader = get_loader(file_path)

            if loader is None:
                continue

            docs = loader.load()

            docs_with_metadata = [
                add_metadata(
                    doc,
                    doc_type,
                    file_path
                )
                for doc in docs
            ]

            loaded_docs.extend(
                docs_with_metadata
            )

        except Exception as e:
            skipped_files.append(
                (
                    file_path,
                    f"Loading error: {str(e)}"
                )
            )

    return loaded_docs, skipped_files