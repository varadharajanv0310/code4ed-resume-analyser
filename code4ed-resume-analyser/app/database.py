"""
Database operations for Resume Relevance Check System
Using SQLite as specified in the tech stack
"""
import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json


class DatabaseManager:
    """Handles all database operations for the system"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.db_path = self.project_root / "database" / "resume_relevance.db"

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enables dict-like access to rows
        return conn

    # =============== JOB DESCRIPTIONS OPERATIONS ===============

    def insert_job_description(self, title: str, company: str, description: str,
                               requirements: str = "", skills_required: str = "",
                               experience_level: str = "", location: str = "",
                               uploaded_by: str = "placement_team") -> int:
        """Insert a new job description"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO job_descriptions
                           (title, company, description, requirements, skills_required,
                            experience_level, location, uploaded_by)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           """, (title, company, description, requirements, skills_required,
                                 experience_level, location, uploaded_by))
            return cursor.lastrowid

    def get_job_descriptions(self, active_only: bool = True) -> List[Dict]:
        """Get all job descriptions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                    SELECT * \
                    FROM job_descriptions
                    WHERE is_active = ? \
                       OR ? = False
                    ORDER BY uploaded_at DESC \
                    """
            cursor.execute(query, (active_only, active_only))
            return [dict(row) for row in cursor.fetchall()]

    def get_job_description_by_id(self, job_id: int) -> Optional[Dict]:
        """Get specific job description by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM job_descriptions WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # =============== RESUMES OPERATIONS ===============

    def insert_resume(self, student_name: str, student_email: str, file_name: str,
                      file_path: str, file_type: str, extracted_text: str = "",
                      skills_extracted: str = "", experience_years: int = 0,
                      education_level: str = "", file_size: int = 0) -> int:
        """Insert a new resume"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO resumes
                           (student_name, student_email, file_name, file_path, file_type,
                            extracted_text, skills_extracted, experience_years,
                            education_level, file_size)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """, (student_name, student_email, file_name, file_path, file_type,
                                 extracted_text, skills_extracted, experience_years,
                                 education_level, file_size))
            return cursor.lastrowid

    def get_resumes(self) -> List[Dict]:
        """Get all resumes"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM resumes ORDER BY uploaded_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_resume_by_id(self, resume_id: int) -> Optional[Dict]:
        """Get specific resume by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # =============== EVALUATIONS OPERATIONS ===============

    def insert_evaluation(self, resume_id: int, job_id: int, relevance_score: int,
                          keyword_score: float, semantic_score: float, verdict: str,
                          missing_skills: str = "", feedback: str = "",
                          evaluation_time_seconds: float = 0) -> int:
        """Insert a new evaluation result"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO evaluations
                           (resume_id, job_id, relevance_score, keyword_score, semantic_score,
                            verdict, missing_skills, feedback, evaluation_time_seconds)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """, (resume_id, job_id, relevance_score, keyword_score, semantic_score,
                                 verdict, missing_skills, feedback, evaluation_time_seconds))
            return cursor.lastrowid

    def get_evaluations_by_job(self, job_id: int) -> List[Dict]:
        """Get all evaluations for a specific job"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT e.*,
                                  r.student_name,
                                  r.student_email,
                                  r.file_name,
                                  j.title as job_title,
                                  j.company
                           FROM evaluations e
                                    JOIN resumes r ON e.resume_id = r.id
                                    JOIN job_descriptions j ON e.job_id = j.id
                           WHERE e.job_id = ?
                           ORDER BY e.relevance_score DESC
                           """, (job_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_evaluations_by_resume(self, resume_id: int) -> List[Dict]:
        """Get all evaluations for a specific resume"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT e.*, j.title as job_title, j.company
                           FROM evaluations e
                                    JOIN job_descriptions j ON e.job_id = j.id
                           WHERE e.resume_id = ?
                           ORDER BY e.relevance_score DESC
                           """, (resume_id,))
            return [dict(row) for row in cursor.fetchall()]

    def check_evaluation_exists(self, resume_id: int, job_id: int) -> bool:
        """Check if evaluation already exists for resume-job pair"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT COUNT(*)
                           FROM evaluations
                           WHERE resume_id = ?
                             AND job_id = ?
                           """, (resume_id, job_id))
            return cursor.fetchone()[0] > 0

    # =============== AUDIT LOGS OPERATIONS ===============

    def log_action(self, action: str, table_name: str = "", record_id: int = None,
                   details: str = "", user_info: str = "system") -> int:
        """Log system actions for audit"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO audit_logs (action, table_name, record_id, details, user_info)
                           VALUES (?, ?, ?, ?, ?)
                           """, (action, table_name, record_id, details, user_info))
            return cursor.lastrowid

    # =============== ANALYTICS OPERATIONS ===============

    def get_system_stats(self) -> Dict:
        """Get system statistics for dashboard"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total counts
            cursor.execute("SELECT COUNT(*) FROM resumes")
            total_resumes = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM job_descriptions WHERE is_active = 1")
            active_jobs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_evaluations = cursor.fetchone()[0]

            # Average scores
            cursor.execute("SELECT AVG(relevance_score) FROM evaluations")
            avg_score = cursor.fetchone()[0] or 0

            # High/Medium/Low verdicts count
            cursor.execute("""
                           SELECT verdict, COUNT(*)
                           FROM evaluations
                           GROUP BY verdict
                           """)
            verdict_counts = dict(cursor.fetchall())

            return {
                'total_resumes': total_resumes,
                'active_jobs': active_jobs,
                'total_evaluations': total_evaluations,
                'average_score': round(avg_score, 2),
                'verdict_counts': verdict_counts
            }


# Global database instance
db_manager = DatabaseManager()