from app.services.keyword_matcher import keyword_matcher

print("Keyword matcher loaded successfully!")

# Test data
resume_text = """
I have 3 years experience in Python development using Django and Flask frameworks. 
I also have experience with React, JavaScript, SQL databases, and REST APIs.
I have worked on machine learning projects using pandas and scikit-learn.
"""

jd_text = """
We are looking for a Python developer with experience in Django framework.
Required skills: Python, Django, SQL, REST APIs, JavaScript.
Experience with React and machine learning is a plus.
"""

resume_skills = {'python', 'django', 'flask', 'react', 'javascript', 'sql', 'machine learning', 'pandas'}
jd_skills = {'python', 'django', 'sql', 'rest apis', 'javascript', 'react', 'machine learning'}

# Test comprehensive matching
result = keyword_matcher.comprehensive_keyword_matching(
    resume_text, jd_text, resume_skills, jd_skills
)

print(f"\nMatching Results:")
print(f"Combined Score: {result['combined_percentage']}%")
print(f"TF-IDF Similarity: {result['tfidf_matching']['similarity_percentage']}%")
print(f"BM25 Score: {result['bm25_matching']['bm25_percentage']}%")
print(f"Exact Match: {result['exact_matching']['match_percentage']}%")
print(f"Fuzzy Skills: {result['fuzzy_skill_matching']['match_percentage']}%")