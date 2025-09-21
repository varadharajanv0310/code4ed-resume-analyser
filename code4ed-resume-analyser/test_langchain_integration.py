from app.services.langchain_pipeline import resume_pipeline
from app.services.langgraph_workflow import workflow_graph

print("LangChain integration loaded successfully!")

# Test data
sample_resume = """
I am a Python developer with 3 years of experience in Django and Flask frameworks.
I have worked on machine learning projects using scikit-learn and pandas.
Experience with React and JavaScript for frontend development.
Bachelor's degree in Computer Science.
"""

sample_jd = """
We are looking for a Python Developer with 2-3 years of experience.
Required: Python, Django, SQL, REST APIs
Nice to have: Machine Learning, JavaScript, React
Bachelor's degree in Computer Science or related field required.
"""

print("\n1. Testing LangChain Pipeline...")
pipeline_result = resume_pipeline.process_resume_evaluation(
    resume_text=sample_resume,
    jd_text=sample_jd,
    resume_id="test_resume_pipeline",
    job_id="test_job_pipeline"
)

print(f"Pipeline Status: {pipeline_result['pipeline_status']}")
print(f"Final Score: {pipeline_result['final_scores']['overall_score']}/100")
print(f"Verdict: {pipeline_result['final_scores']['verdict']}")
print(f"Processing Time: {pipeline_result['processing_time_seconds']:.2f}s")

print("\n2. Testing LangGraph Workflow...")
workflow_result = workflow_graph.execute_workflow(
    resume_text=sample_resume,
    jd_text=sample_jd,
    resume_id="test_resume_workflow",
    job_id="test_job_workflow"
)

print(f"Workflow Status: {workflow_result['workflow_status']}")
print(f"Final Score: {workflow_result['final_scores']['overall_score']}/100")
print(f"Verdict: {workflow_result['final_scores']['verdict']}")
print(f"Processing Time: {workflow_result['processing_time_seconds']:.2f}s")
print(f"Steps Executed: {' -> '.join(workflow_result['step_history'])}")

if workflow_result['errors']:
    print(f"Errors: {workflow_result['errors']}")

print("\n3. Testing Pipeline Stats...")
pipeline_stats = resume_pipeline.get_pipeline_stats()
print(f"Component Status: {pipeline_stats['components_status']}")
print(f"LangChain Chains: {pipeline_stats['langchain_chains']}")

print("\nLangChain integration test completed!")