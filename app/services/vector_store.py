"""
Vector store service using Chroma (approved package)
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import uuid


class VectorStore:
    """Chroma vector database for storing and retrieving embeddings"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Setup Chroma database path
        self.db_path = Path(__file__).parent.parent.parent / "data" / "vector_db"
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Default embedding function (sentence transformers)
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

            # Collections for different document types
            self.resume_collection = None
            self.jd_collection = None

            self._initialize_collections()

        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma: {str(e)}")
            self.client = None

    def _initialize_collections(self):
        """Initialize Chroma collections for resumes and job descriptions"""
        try:
            # Resume collection
            self.resume_collection = self.client.get_or_create_collection(
                name="resumes",
                embedding_function=self.embedding_function,
                metadata={"description": "Resume documents for similarity matching"}
            )

            # Job Description collection
            self.jd_collection = self.client.get_or_create_collection(
                name="job_descriptions",
                embedding_function=self.embedding_function,
                metadata={"description": "Job description documents for similarity matching"}
            )

            self.logger.info("Chroma collections initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize collections: {str(e)}")

    def add_resume(self, resume_id: str, text: str, metadata: Dict = None) -> bool:
        """Add resume to vector store"""
        if not self.resume_collection:
            self.logger.error("Resume collection not initialized")
            return False

        try:
            # Prepare metadata
            doc_metadata = {
                "resume_id": resume_id,
                "document_type": "resume",
                "text_length": len(text)
            }
            if metadata:
                doc_metadata.update(metadata)

            # Add to collection
            self.resume_collection.add(
                documents=[text],
                ids=[f"resume_{resume_id}"],
                metadatas=[doc_metadata]
            )

            self.logger.info(f"Added resume {resume_id} to vector store")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add resume {resume_id}: {str(e)}")
            return False

    def add_job_description(self, job_id: str, text: str, metadata: Dict = None) -> bool:
        """Add job description to vector store"""
        if not self.jd_collection:
            self.logger.error("Job description collection not initialized")
            return False

        try:
            # Prepare metadata
            doc_metadata = {
                "job_id": job_id,
                "document_type": "job_description",
                "text_length": len(text)
            }
            if metadata:
                doc_metadata.update(metadata)

            # Add to collection
            self.jd_collection.add(
                documents=[text],
                ids=[f"job_{job_id}"],
                metadatas=[doc_metadata]
            )

            self.logger.info(f"Added job description {job_id} to vector store")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add job description {job_id}: {str(e)}")
            return False

    def semantic_search_resumes(self, query_text: str, n_results: int = 10) -> List[Dict]:
        """Search similar resumes using semantic similarity"""
        if not self.resume_collection:
            return []

        try:
            results = self.resume_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                })

            return formatted_results

        except Exception as e:
            self.logger.error(f"Semantic search failed: {str(e)}")
            return []

    def semantic_search_jobs(self, query_text: str, n_results: int = 10) -> List[Dict]:
        """Search similar job descriptions using semantic similarity"""
        if not self.jd_collection:
            return []

        try:
            results = self.jd_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]
                })

            return formatted_results

        except Exception as e:
            self.logger.error(f"Semantic search failed: {str(e)}")
            return []

    def calculate_resume_jd_similarity(self, resume_id: str, job_id: str) -> float:
        """Calculate semantic similarity between specific resume and job description"""
        if not self.resume_collection or not self.jd_collection:
            return 0.0

        try:
            # Get resume document
            resume_result = self.resume_collection.get(
                ids=[f"resume_{resume_id}"],
                include=["documents"]
            )

            # Get job description document
            jd_result = self.jd_collection.get(
                ids=[f"job_{job_id}"],
                include=["documents"]
            )

            if not resume_result['documents'] or not jd_result['documents']:
                return 0.0

            resume_text = resume_result['documents'][0]
            jd_text = jd_result['documents'][0]

            # Use job description to search similar resumes
            similar_resumes = self.resume_collection.query(
                query_texts=[jd_text],
                n_results=100,  # Search more to find our specific resume
                include=["metadatas", "distances"]
            )

            # Find our specific resume in results
            for i, metadata in enumerate(similar_resumes['metadatas'][0]):
                if metadata.get('resume_id') == resume_id:
                    distance = similar_resumes['distances'][0][i]
                    similarity = 1 - distance
                    return max(0.0, min(1.0, similarity))  # Clamp between 0 and 1

            return 0.0

        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {str(e)}")
            return 0.0

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about stored documents"""
        stats = {
            'total_resumes': 0,
            'total_jobs': 0
        }

        try:
            if self.resume_collection:
                stats['total_resumes'] = self.resume_collection.count()

            if self.jd_collection:
                stats['total_jobs'] = self.jd_collection.count()

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")

        return stats

    def delete_resume(self, resume_id: str) -> bool:
        """Delete resume from vector store"""
        if not self.resume_collection:
            return False

        try:
            self.resume_collection.delete(ids=[f"resume_{resume_id}"])
            self.logger.info(f"Deleted resume {resume_id} from vector store")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete resume {resume_id}: {str(e)}")
            return False

    def delete_job(self, job_id: str) -> bool:
        """Delete job description from vector store"""
        if not self.jd_collection:
            return False

        try:
            self.jd_collection.delete(ids=[f"job_{job_id}"])
            self.logger.info(f"Deleted job {job_id} from vector store")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete job {job_id}: {str(e)}")
            return False

    def reset_collections(self) -> bool:
        """Reset all collections (for testing/development)"""
        try:
            if self.client:
                self.client.reset()
                self._initialize_collections()
                self.logger.info("Vector store collections reset successfully")
                return True
        except Exception as e:
            self.logger.error(f"Failed to reset collections: {str(e)}")
        return False


# Global vector store instance
vector_store = VectorStore()