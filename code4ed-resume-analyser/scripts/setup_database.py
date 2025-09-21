"""
Database setup script for Resume Relevance Check System
"""
import sqlite3
import os
from pathlib import Path


def setup_database():
    """Initialize the SQLite database with required tables"""

    # Get project root directory
    project_root = Path(__file__).parent.parent
    database_dir = project_root / "database"
    database_path = database_dir / "resume_relevance.db"
    schema_path = database_dir / "schema.sql"

    # Create database directory if it doesn't exist
    database_dir.mkdir(exist_ok=True)

    # Create data directories
    data_dir = project_root / "data"
    (data_dir / "uploads" / "resumes").mkdir(parents=True, exist_ok=True)
    (data_dir / "uploads" / "job_descriptions").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed" / "extracted_texts").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed" / "embeddings").mkdir(parents=True, exist_ok=True)
    (data_dir / "vector_db").mkdir(parents=True, exist_ok=True)
    (data_dir / "models").mkdir(parents=True, exist_ok=True)

    try:
        # Connect to database (creates file if doesn't exist)
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Read and execute schema
        with open(schema_path, 'r', encoding='utf-8') as schema_file:
            schema_sql = schema_file.read()
            cursor.executescript(schema_sql)

        # Insert sample job description for testing
        sample_job = """
                     INSERT \
                     OR IGNORE INTO job_descriptions 
        (id, title, company, description, requirements, skills_required, experience_level, location, uploaded_by)
        VALUES 
        (1, 'Python Developer', 'Innomatics Research Labs', 
         'We are looking for a skilled Python developer to join our team.',
         'Bachelor degree in Computer Science, 2+ years Python experience',
         'Python, Django, FastAPI, SQL, Git, REST APIs', 
         'Mid-level', 'Hyderabad', 'placement_team') \
                     """
        cursor.execute(sample_job)

        conn.commit()
        conn.close()

        print(f"✅ Database setup completed successfully!")
        print(f"📁 Database location: {database_path}")
        print(f"📁 Data directories created under: {data_dir}")

        return True

    except Exception as e:
        print(f"❌ Error setting up database: {str(e)}")
        return False


if __name__ == "__main__":
    setup_database()