from app.services.vector_store import vector_store

print("Vector store loaded successfully!")

# Test data
sample_resume = """
I am a Python developer with 3 years of experience in Django and Flask frameworks.
I have worked on machine learning projects using scikit-learn and pandas.
Experience with React and JavaScript for frontend development.
"""

sample_jd = """
We are looking for a Python developer with Django experience.
Required: Python, Django, SQL, REST APIs
Nice to have: React, Machine Learning, JavaScript
"""

# Test adding documents
print("\nTesting vector store operations...")

# Add resume
success1 = vector_store.add_resume(
    resume_id="test_resume_1",
    text=sample_resume,
    metadata={"student_name": "Test Student", "experience_years": 3}
)

# Add job description
success2 = vector_store.add_job_description(
    job_id="test_job_1",
    text=sample_jd,
    metadata={"company": "Test Company", "position": "Python Developer"}
)

print(f"Resume added: {success1}")
print(f"Job description added: {success2}")

# Get collection statistics
stats = vector_store.get_collection_stats()
print(f"Collection stats: {stats}")

# Test similarity calculation
if success1 and success2:
    similarity = vector_store.calculate_resume_jd_similarity("test_resume_1", "test_job_1")
    print(f"Semantic similarity: {similarity:.4f} ({similarity*100:.2f}%)")

# Test semantic search
print("\nTesting semantic search...")
search_results = vector_store.semantic_search_resumes("Python Django developer", n_results=5)
print(f"Found {len(search_results)} similar resumes")

if search_results:
    for result in search_results[:2]:  # Show top 2
        print(f"  - ID: {result['id']}, Similarity: {result['similarity_score']:.4f}")

print("\nVector store test completed!")