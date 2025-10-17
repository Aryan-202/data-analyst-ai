import os
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data_analyst_ai.db")

    # File Storage
    upload_folder: str = os.getenv("UPLOAD_FOLDER", "data/raw")
    processed_folder: str = os.getenv("PROCESSED_FOLDER", "data/processed")
    reports_folder: str = os.getenv("REPORTS_FOLDER", "data/reports")

    # App Settings
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", 100))  # MB
    allowed_file_types: list = ["csv", "xlsx", "xls", "json"]

    # AI Model Settings
    default_llm_model: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
    max_insight_tokens: int = int(os.getenv("MAX_INSIGHT_TOKENS", 500))

    # AutoML Settings
    max_training_time: int = int(os.getenv("MAX_TRAINING_TIME", 300))  # seconds

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()