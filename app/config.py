from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("storage")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    min_score: float = 0.25

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    ocr_lang: str = "eng"
    ocr_min_chars_per_page: int = 40


settings = Settings()
