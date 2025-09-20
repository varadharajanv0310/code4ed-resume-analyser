-- Resume Relevance Check System Database Schema

-- Table for storing job descriptions
CREATE TABLE IF NOT EXISTS job_descriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title VARCHAR(255) NOT NULL,
    company VARCHAR(255),
    description TEXT NOT NULL,
    requirements TEXT,
    skills_required TEXT,
    experience_level VARCHAR(50),
    location VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_by VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE
);

-- Table for storing resume information
CREATE TABLE IF NOT EXISTS resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name VARCHAR(255),
    student_email VARCHAR(255),
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    extracted_text TEXT,
    skills_extracted TEXT,
    experience_years INTEGER,
    education_level VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER
);

-- Table for storing evaluation results
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    relevance_score INTEGER NOT NULL,
    keyword_score REAL,
    semantic_score REAL,
    verdict VARCHAR(20) NOT NULL,
    missing_skills TEXT,
    feedback TEXT,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluation_time_seconds REAL,
    FOREIGN KEY (resume_id) REFERENCES resumes (id),
    FOREIGN KEY (job_id) REFERENCES job_descriptions (id)
);

-- Table for storing system audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action VARCHAR(100) NOT NULL,
    table_name VARCHAR(50),
    record_id INTEGER,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_info VARCHAR(255)
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_evaluations_resume_id ON evaluations(resume_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_job_id ON evaluations(job_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_score ON evaluations(relevance_score);
CREATE INDEX IF NOT EXISTS idx_resumes_uploaded_at ON resumes(uploaded_at);
CREATE INDEX IF NOT EXISTS idx_jobs_active ON job_descriptions(is_active);