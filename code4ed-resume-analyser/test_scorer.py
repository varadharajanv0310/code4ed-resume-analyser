from app.services.scorer import relevance_scorer

print("Relevance scorer loaded successfully!")

# Test data simulating results from keyword and semantic matching
sample_keyword_results = {
    'exact_matching': {'match_percentage': 70},
    'tfidf_matching': {'similarity_score': 0.75},
    'bm25_matching': {'bm25_score': 0.68},
    'fuzzy_skill_matching': {
        'match_percentage': 80,
        'missing_skills': {'docker', 'kubernetes', 'aws'}
    }
}

sample_semantic_results = {
    'semantic_score': 78,
    'recommendation': 'interview'
}

# Test comprehensive scoring
print("\nTesting comprehensive scoring...")
score_result = relevance_scorer.calculate_comprehensive_score(
    keyword_results=sample_keyword_results,
    semantic_results=sample_semantic_results,
    vector_similarity=0.72,
    resume_experience=3,
    required_experience=2,
    resume_education="bachelors",
    required_education="bachelors"
)

print(f"Overall Score: {score_result['overall_score']}/100")
print(f"Verdict: {score_result['verdict']}")
print(f"Component Scores: {score_result['component_scores']}")
print(f"Missing Skills: {score_result['missing_skills']}")
print(f"Recommendations: {score_result['recommendations']}")

# Test batch scoring
print("\nTesting batch scoring...")
batch_data = [
    {
        'resume_id': 'resume_1',
        'keyword_results': sample_keyword_results,
        'semantic_results': sample_semantic_results,
        'vector_similarity': 0.85,
        'resume_experience': 4,
        'required_experience': 3
    },
    {
        'resume_id': 'resume_2',
        'keyword_results': {
            'exact_matching': {'match_percentage': 45},
            'tfidf_matching': {'similarity_score': 0.42}
        },
        'semantic_results': {'semantic_score': 40},
        'resume_experience': 1,
        'required_experience': 3
    }
]

batch_results = relevance_scorer.batch_score_resumes(batch_data)
print(f"Batch Results:")
for result in batch_results:
    print(f"  {result['resume_id']}: {result['overall_score']}/100 ({result['verdict']})")

print("\nScoring system test completed!")