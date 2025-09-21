"""
Vector store service - Streamlit Cloud compatible version
ChromaDB disabled due to SQLite version conflicts
"""
import logging
from typing import Dict, List, Optional


class VectorStore:
    """Fallback vector store when ChromaDB is not available"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.logger.warning("ChromaDB not available - using fallback mode")

    def add_resume(self, resume_id: str, text: str, metadata: Dict = None) -> bool:
        """Fallback - always returns True"""
        return True

    def add_job_description(self, job_id: str, text: str, metadata: Dict = None) -> bool:
        """Fallback - always returns True"""
        return True

    def calculate_resume_jd_similarity(self, resume_id: str, job_id: str) -> float:
        """Fallback - returns default similarity score"""
        return 0.0

    def semantic_search_resumes(self, query_text: str, n_results: int = 10) -> List[Dict]:
        """Fallback - returns empty list"""
        return []

    def semantic_search_jobs(self, query_text: str, n_results: int = 10) -> List[Dict]:
        """Fallback - returns empty list"""
        return []

    def get_collection_stats(self) -> Dict[str, int]:
        """Fallback - returns zero stats"""
        return {'total_resumes': 0, 'total_jobs': 0}

    def delete_resume(self, resume_id: str) -> bool:
        """Fallback - always returns True"""
        return True

    def delete_job(self, job_id: str) -> bool:
        """Fallback - always returns True"""
        return True

    def reset_collections(self) -> bool:
        """Fallback - always returns True"""
        return True


# Global vector store instance
vector_store = VectorStore()
