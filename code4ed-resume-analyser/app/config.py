"""
Configuration settings for Resume Relevance Check System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"
    RESUMES_DIR = UPLOADS_DIR / "resumes"
    JD_DIR = UPLOADS_DIR / "job_descriptions"
    PROCESSED_DIR = DATA_DIR / "processed"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"

    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

    # Application settings
    APP_NAME = os.getenv("APP_NAME", "Resume Relevance Check System")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

    # File upload settings
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "pdf,docx").split(",")

    # Scoring configuration
    KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.4"))
    SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
    MIN_RELEVANCE_SCORE = int(os.getenv("MIN_RELEVANCE_SCORE", "30"))

    # LangChain settings
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    @classmethod
    def validate_config(cls) -> bool:
        """Validate required configuration"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required in .env file")
        return True


# Validate configuration on import
Config.validate_config()