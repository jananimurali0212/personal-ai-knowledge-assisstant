"""Application settings and configuration."""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from backend/.env or root .env if present
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

class Settings:
    PROJECT_NAME: str = "Personal AI Knowledge Assistant"
    VERSION: str = "0.2.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS Origins for Frontend
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Supabase Configuration (read from environment variables, never hardcoded)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://oeczcuvxbqdbepgzzohk.supabase.co")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    # PostgreSQL Direct Connection (optional for direct asyncpg / migrations)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Gemini Embedding Configuration
    # Supported model: models/text-embedding-004
    # Supported native output dimensionality: 768
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    GEMINI_EMBEDDING_DIMENSIONS: int = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768"))

    # Chunking Configuration
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))  # characters
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))  # characters

settings = Settings()
