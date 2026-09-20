import os
from dotenv import load_dotenv

load_dotenv(override=True)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Please add it to your .env file."
    )


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-oss-120b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_RETRIEVAL_K = 25
MAX_TOTAL_TOKENS = 250000
MAX_CHUNK_TOKENS = 8000