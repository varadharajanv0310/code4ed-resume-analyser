from app.services.text_processor import text_processor

print("Text processor loaded successfully!")

# Test text processing
sample_text = "I have 3 years experience in Python and Django development. I also know React and SQL."
result = text_processor.process_text(sample_text)

print(f"Sample processing:")
print(f"  Skills: {result['skills']}")
print(f"  Experience: {result['experience_years']} years")
print(f"  Education: {result['education_level']}")
print(f"  Keywords: {[kw[0] for kw in result['keywords'][:5]]}")  # Top 5 keywords