from app.services.llm_service import llm_service

print("LLM service loaded successfully!")

# Test API connection
print("Testing Gemini API connection...")
connection_ok = llm_service.test_connection()
print(f"API Connection: {'✅ Working' if connection_ok else '❌ Failed'}")

if connection_ok:
    print("\nTesting LLM functions...")

    # Test data
    sample_resume = """
    I am a Python developer with 3 years of experience. I have worked with Django, Flask, 
    and FastAPI frameworks. I have experience with SQL databases, REST APIs, and some 
    machine learning projects using scikit-learn. I also know JavaScript and React.
    """

    sample_jd = """
    We are looking for a Python Developer with 2-3 years of experience.
    Required: Python, Django, SQL, REST APIs
    Nice to have: Machine Learning, JavaScript, React
    """

    # Test semantic analysis
    print("1. Testing semantic analysis...")
    analysis = llm_service.generate_semantic_analysis(sample_resume, sample_jd)
    if "error" not in analysis:
        print(f"   Semantic Score: {analysis.get('semantic_score', 'N/A')}")
        print(f"   Recommendation: {analysis.get('recommendation', 'N/A')}")
    else:
        print(f"   Error: {analysis['error']}")

    # Test feedback generation
    print("\n2. Testing feedback generation...")
    feedback = llm_service.generate_feedback(sample_resume, sample_jd, 75)
    print(f"   Feedback length: {len(feedback)} characters")
    print(f"   Preview: {feedback[:150]}...")

    # Test job requirements extraction
    print("\n3. Testing job requirements extraction...")
    requirements = llm_service.extract_job_requirements(sample_jd)
    if "error" not in requirements:
        print(f"   Required skills: {requirements.get('required_skills', [])}")
        print(f"   Job level: {requirements.get('job_level', 'N/A')}")
    else:
        print(f"   Error: {requirements['error']}")

    # Test verdict generation
    print("\n4. Testing verdict generation...")
    verdict, reasoning = llm_service.generate_verdict(75, 80, 78)
    print(f"   Verdict: {verdict}")
    print(f"   Reasoning: {reasoning}")

else:
    print("\n❌ Cannot test LLM functions - API connection failed")
    print("Please check your GOOGLE_API_KEY in the .env file")

print("\nLLM service test completed!")