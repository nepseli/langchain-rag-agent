from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str

    # Models
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Storage paths (relative to project root)
    db_path: Path = BASE_DIR / "db" / "rag_agent.db"
    chroma_path: Path = BASE_DIR / "db" / "chroma"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 12
    max_chunks_per_doc: int = 4       # cap per-doc in default retrieval
    top_k_per_doc_comparative: int = 4  # chunks per doc in comparative fan-out


settings = Settings()

# Ensure storage directories exist at import time
settings.db_path.parent.mkdir(parents=True, exist_ok=True)
settings.chroma_path.mkdir(parents=True, exist_ok=True)
