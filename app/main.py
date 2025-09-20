"""
Complete Resume-JD Evaluation System with Hybrid Data Support
Fixed version with no syntax errors
"""
import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import configuration
from app.config import Config

# Set page configuration
st.set_page_config(
    page_title="Resume-JD Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in-up { animation: fadeInUp 0.8s ease-out; }

    .main-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        text-align: center;
    }

    .success-alert {
        background: linear-gradient(45deg, #56CCF2, #2F80ED);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }

    .error-alert {
        background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        font-weight: 500;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)


load_custom_css()


# Initialize session state
def initialize_session_state():
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1


def main():
    initialize_session_state()

    # Header
    st.markdown("""
    <div class="fade-in-up">
        <h1 style="text-align: center; font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🎯 Resume-JD Analyzer
        </h1>
        <p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            AI-Powered Resume Evaluation with Hybrid Data Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main workflow tabs
    tab1, tab2, tab3 = st.tabs(["📋 Step 1: Job Description", "📄 Step 2: Upload Resume", "📊 Step 3: View Results"])

    with tab1:
        show_job_description_input()

    with tab2:
        show_resume_upload()

    with tab3:
        show_evaluation_results()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; color: white; margin-bottom: 2rem;">
            <h2>🎯 Quick Actions</h2>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📋 Load Sample JD", use_container_width=True):
            load_sample_job_description()

        if st.button("🔄 Reset All", use_container_width=True):
            reset_application()

        st.markdown("---")

        # System status
        st.markdown("### System Status")
        try:
            from app.services.llm_service import enhanced_llm_service

            llm_status = enhanced_llm_service.test_connection()

            if llm_status['status'] == 'success':
                st.markdown("🤖 **LLM Service:** ✅ Available")
                st.markdown(f"📊 **Quota Remaining:** {llm_status.get('remaining_quota', 'Unknown')}")
            elif llm_status['status'] == 'quota_exceeded':
                st.markdown("🤖 **LLM Service:** ⚠️ Quota Exceeded")
                st.markdown("🔧 **Hybrid Fallback:** ✅ Active")
            else:
                st.markdown("🤖 **LLM Service:** ❌ Unavailable")
                st.markdown("🔧 **Hybrid Fallback:** ✅ Active")

        except Exception as e:
            st.error(f"System check failed: {str(e)}")


def show_job_description_input():
    """Step 1: Job Description Input"""
    st.markdown("""
    <div class="main-card">
        <h2>📋 Step 1: Enter Job Description</h2>
        <p>Paste the complete job description that you want to evaluate resumes against</p>
    </div>
    """, unsafe_allow_html=True)

    job_description = st.text_area(
        "Job Description",
        value=st.session_state.job_description,
        height=400,
        placeholder="""Paste your complete job description here...

Example:
1. Data Science Interns
• Internship Duration: 6 months, followed by permanent employment
• Role Overview: Work on data engineering, data visualization, and data science tasks
• Build deep learning models using Generative AI, Computer Vision, and NLP
• Eligibility: B.Tech, BE, 2023 and earlier pass-outs
• Skills: Python, Machine Learning, Deep Learning, Pandas, SQL
• Location: Pune (Onsite)
• Stipend: ₹5,000 per month
        """,
        help="Include all relevant details: role requirements, skills, experience, education, location, etc."
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("📝 Save Job Description", use_container_width=True, type="primary"):
            if job_description.strip():
                st.session_state.job_description = job_description
                st.session_state.current_step = 2

                st.markdown("""
                <div class="success-alert">
                    ✅ Job description saved! Now proceed to Step 2 to upload resume.
                </div>
                """, unsafe_allow_html=True)

                try:
                    from app.database import db_manager
                    job_id = db_manager.insert_job_description(
                        title="Custom Job Evaluation",
                        company="Evaluation Session",
                        description=job_description,
                        uploaded_by="User"
                    )
                    st.session_state.current_job_id = job_id
                except Exception as e:
                    st.warning(f"Could not save to database: {str(e)}")

            else:
                st.error("Please enter a job description before proceeding.")

    if job_description.strip():
        with st.expander("🤖 AI Analysis Preview", expanded=False):
            st.info("💡 **Tip:** The AI will analyze resumes for these key aspects:")

            try:
                from app.services.text_processor import text_processor
                processed = text_processor.process_text(job_description)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🔍 Detected Skills:**")
                    skills = list(processed['skills'])[:10]
                    for skill in skills:
                        st.markdown(f"• {skill}")

                with col2:
                    st.markdown("**📊 Key Requirements:**")
                    keywords = processed['keywords'][:5]
                    for keyword, score in keywords:
                        st.markdown(f"• {keyword} ({score:.2f})")

            except Exception as e:
                st.warning(f"Preview analysis failed: {str(e)}")


def show_resume_upload():
    """Step 2: Resume Upload with Real-Time Magic"""
    if not st.session_state.job_description:
        st.warning("⚠️ Please complete Step 1 first - enter a job description.")
        return

    st.markdown("""
    <div class="main-card">
        <h2>📄 Step 2: Upload Resume for Real-Time Analysis</h2>
        <p>Upload the resume file and watch AI extract information in real-time!</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose resume file",
        type=['pdf', 'docx'],
        help="Supported formats: PDF, DOCX (Max size: 10MB)"
    )

    if uploaded_file:
        # Real-time file analysis
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="success-alert">
                ✅ File uploaded! Processing in real-time...
            </div>
            """, unsafe_allow_html=True)

            # Real-time parsing
            with st.spinner("🔍 Extracting content..."):
                parsed_content = perform_realtime_parsing(uploaded_file)

            if parsed_content['success']:
                # Live extraction preview
                st.markdown("### 📄 Live Extraction Preview")

                # Animated text extraction
                with st.container():
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; max-height: 300px; overflow-y: auto;">
                        <h5 style="color: #667eea; margin-top: 0;">🔤 Extracted Text</h5>
                    """, unsafe_allow_html=True)

                    # Show first 500 characters with typing animation
                    text_preview = parsed_content['text'][:500] + "..." if len(parsed_content['text']) > 500 else \
                    parsed_content['text']

                    # Create typing effect container
                    text_container = st.empty()

                    # Simulate typing animation
                    typing_text = ""
                    for i, char in enumerate(text_preview):
                        typing_text += char
                        if i % 10 == 0:  # Update every 10 characters
                            text_container.markdown(f"```\n{typing_text}\n```")
                            import time
                            time.sleep(0.02)  # Small delay for animation

                    st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            if parsed_content['success']:
                # Live skill extraction with highlighting
                st.markdown("### 🎯 Live Skill Detection")

                skills_detected = parsed_content.get('skills', set())

                if skills_detected:
                    # Animated skill detection
                    skill_container = st.empty()

                    # Show skills appearing one by one
                    displayed_skills = []
                    for i, skill in enumerate(list(skills_detected)[:15]):  # Show max 15
                        displayed_skills.append(skill)

                        # Create animated skill display
                        skills_html = ""
                        for j, s in enumerate(displayed_skills):
                            if j == len(displayed_skills) - 1:  # Latest skill
                                skills_html += f'<span style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; margin: 0.2rem; display: inline-block; animation: fadeInUp 0.5s ease-out;">🔥 {s}</span>'
                            else:
                                skills_html += f'<span style="background: #e3f2fd; color: #1976d2; padding: 0.3rem 0.8rem; border-radius: 20px; margin: 0.2rem; display: inline-block;">✅ {s}</span>'

                        skill_container.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px; border: 2px solid #e0e0e0;">
                            <h6 style="color: #667eea; margin-bottom: 0.5rem;">Skills Found: {len(displayed_skills)}</h6>
                            {skills_html}
                        </div>
                        """, unsafe_allow_html=True)

                        import time
                        time.sleep(0.3)  # Delay between skill appearances

                # Experience and education detection
                st.markdown("### 📊 Profile Analysis")

                experience_years = parsed_content.get('experience_years', 0)
                education_level = parsed_content.get('education_level', 'not_specified')

                # Animated metrics
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #4CAF50, #45a049); border-radius: 10px; color: white; animation: pulse 2s infinite;">
                        <h3 style="margin: 0; font-size: 2rem;">{experience_years}</h3>
                        <p style="margin: 0;">Years Experience</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_b:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2196F3, #1976D2); border-radius: 10px; color: white; animation: pulse 2s infinite;">
                        <h3 style="margin: 0; font-size: 1.2rem;">{education_level.replace('_', ' ').title()}</h3>
                        <p style="margin: 0;">Education Level</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Real-time compatibility check
                st.markdown("### ⚡ Real-Time Compatibility Check")

                # Quick compatibility analysis
                jd_skills = extract_jd_skills_quick(st.session_state.job_description)
                resume_skills = skills_detected

                if jd_skills and resume_skills:
                    matched_skills = jd_skills.intersection(resume_skills)
                    compatibility_score = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0

                    # Animated compatibility meter
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px; border: 2px solid #e0e0e0;">
                        <h6 style="color: #667eea; margin-bottom: 0.5rem;">Quick Compatibility: {compatibility_score:.0f}%</h6>
                        <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; height: 20px;">
                            <div style="width: {compatibility_score}%; background: linear-gradient(45deg, #667eea, #764ba2); height: 20px; border-radius: 10px; transition: width 2s ease-in-out;"></div>
                        </div>
                        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">Matched Skills: {len(matched_skills)}/{len(jd_skills)}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Enhanced candidate information form
        st.markdown("---")
        with st.expander("👤 Candidate Information (Hybrid Intelligence)", expanded=True):
            st.info(
                "💡 **AI Pre-filled:** We've extracted information from the resume. You can verify or correct it below.")

            col1, col2 = st.columns(2)

            with col1:
                # Pre-fill with extracted name if possible
                extracted_name = extract_candidate_name(parsed_content['text']) if parsed_content['success'] else ""
                candidate_name = st.text_input(
                    "Candidate Name",
                    value=extracted_name,
                    placeholder="Enter candidate name"
                )

                candidate_email = st.text_input("Email", placeholder="candidate@email.com")

            with col2:
                # Pre-fill with extracted experience
                experience_years = st.number_input(
                    "Years of Experience",
                    min_value=0,
                    max_value=20,
                    value=parsed_content.get('experience_years', 0) if parsed_content['success'] else 0,
                    help="🤖 AI detected experience from resume. Adjust if needed."
                )

                # Pre-fill with extracted education
                detected_education = parsed_content.get('education_level', 'bachelor\'s') if parsed_content[
                    'success'] else 'bachelor\'s'
                education_options = ["High School", "Diploma", "Bachelor's", "Master's", "PhD"]
                default_index = 2  # Bachelor's as default

                try:
                    if 'master' in detected_education.lower():
                        default_index = 3
                    elif 'phd' in detected_education.lower() or 'doctorate' in detected_education.lower():
                        default_index = 4
                    elif 'diploma' in detected_education.lower():
                        default_index = 1
                    elif 'high school' in detected_education.lower():
                        default_index = 0
                except:
                    default_index = 2

                education_level = st.selectbox(
                    "Education Level",
                    education_options,
                    index=default_index,
                    help="🤖 AI detected education from resume. Verify if correct."
                )

        # Enhanced evaluation button
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("🚀 Complete AI Analysis", use_container_width=True, type="primary"):
                st.markdown("""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); border-radius: 10px; color: white; margin: 1rem 0;">
                    <h4 style="margin: 0;">🎯 Launching Deep AI Analysis...</h4>
                    <p style="margin: 0;">This will take the real-time analysis to the next level!</p>
                </div>
                """, unsafe_allow_html=True)

                evaluate_resume_against_jd_hybrid(
                    uploaded_file,
                    st.session_state.job_description,
                    candidate_name or "Anonymous Candidate",
                    candidate_email or "",
                    experience_years,
                    education_level
                )


def perform_realtime_parsing(uploaded_file):
    """Perform real-time parsing of uploaded resume"""
    try:
        from app.services.file_parser import file_parser
        from app.services.text_processor import text_processor
        from app.config import Config

        # Save file temporarily
        resume_dir = Config.RESUMES_DIR
        resume_dir.mkdir(parents=True, exist_ok=True)

        file_path = resume_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Parse the file
        parse_result = file_parser.parse_file(str(file_path))

        if not parse_result['metadata']['success']:
            return {'success': False, 'error': parse_result['metadata']['error']}

        # Process the text
        resume_text = parse_result['text']
        processed = text_processor.process_text(resume_text)

        return {
            'success': True,
            'text': resume_text,
            'skills': processed['skills'],
            'experience_years': processed['experience_years'],
            'education_level': processed['education_level'],
            'processed_data': processed
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def extract_jd_skills_quick(jd_text):
    """Quick skill extraction from job description for real-time analysis"""
    try:
        from app.services.text_processor import text_processor
        processed = text_processor.process_text(jd_text)
        return processed['skills']
    except:
        # Fallback: simple keyword extraction
        common_skills = {'python', 'java', 'javascript', 'sql', 'react', 'django', 'flask',
                         'machine learning', 'data science', 'ai', 'pandas', 'numpy', 'tensorflow',
                         'aws', 'azure', 'docker', 'kubernetes', 'git', 'html', 'css'}

        found_skills = set()
        jd_lower = jd_text.lower()
        for skill in common_skills:
            if skill in jd_lower:
                found_skills.add(skill)

        return found_skills


def extract_candidate_name(resume_text):
    """Quick name extraction from resume text"""
    try:
        lines = resume_text.split('\n')[:5]  # Check first 5 lines
        for line in lines:
            line = line.strip()
            # Simple heuristic: find line with 2-4 words, proper case, no numbers
            words = line.split()
            if (2 <= len(words) <= 4 and
                    line.replace(' ', '').isalpha() and
                    any(word[0].isupper() for word in words) and
                    len(line) < 50):
                return line
        return ""
    except:
        return ""

def evaluate_resume_against_jd_hybrid(uploaded_file, job_description, candidate_name, candidate_email, experience_years,
                                      education_level):
    """Core function: Evaluate resume against job description with hybrid approach"""

    with st.spinner("🔍 Analyzing resume with hybrid data intelligence... This may take a few seconds."):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("📄 Extracting text from resume...")
            progress_bar.progress(0.2)

            from app.services.file_parser import file_parser
            from app.config import Config

            resume_dir = Config.RESUMES_DIR
            resume_dir.mkdir(parents=True, exist_ok=True)

            file_path = resume_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            parse_result = file_parser.parse_file(str(file_path))

            if not parse_result['metadata']['success']:
                st.error(f"Failed to parse resume: {parse_result['metadata']['error']}")
                return

            resume_text = parse_result['text']

            status_text.text("🧠 Processing and analyzing content...")
            progress_bar.progress(0.4)

            status_text.text("⚙️ Running AI evaluation with hybrid data analysis...")
            progress_bar.progress(0.6)

            from app.services.langgraph_workflow import workflow_graph

            evaluation_result = workflow_graph.execute_workflow(
                resume_text=resume_text,
                jd_text=job_description,
                resume_id=f"eval_{uploaded_file.name}",
                job_id="custom_jd_evaluation",
                manual_experience=experience_years,
                manual_education=education_level.lower()
            )

            status_text.text("📊 Generating detailed hybrid analysis...")
            progress_bar.progress(0.8)

            status_text.text("✅ Hybrid evaluation completed!")
            progress_bar.progress(1.0)

            st.session_state.evaluation_results = {
                'candidate_name': candidate_name,
                'candidate_email': candidate_email,
                'experience_years': experience_years,
                'education_level': education_level,
                'file_name': uploaded_file.name,
                'evaluation_data': evaluation_result
            }

            st.session_state.current_step = 3

            hybrid_data = evaluation_result.get('hybrid_data', {})
            if hybrid_data:
                final_exp = hybrid_data.get('experience_years', experience_years)
                exp_source = hybrid_data.get('experience_source', 'manual')
                final_edu = hybrid_data.get('education_level', education_level)
                edu_source = hybrid_data.get('education_source', 'manual')

                extracted_exp = hybrid_data.get('extracted_experience', 0)
                extracted_edu = hybrid_data.get('extracted_education', 'not_specified')

                st.markdown(f"""
                <div class="success-alert">
                    🎉 <strong>Hybrid Evaluation Completed Successfully!</strong><br><br>
                    📊 <strong>Data Resolution:</strong><br>
                    • <strong>Experience:</strong> Used {final_exp} years ({exp_source})<br>
                    &nbsp;&nbsp;→ Resume showed: {extracted_exp} years | You entered: {experience_years} years<br>
                    • <strong>Education:</strong> Used {final_edu.replace('_', ' ').title()} ({edu_source})<br>
                    &nbsp;&nbsp;→ Resume showed: {extracted_edu.replace('_', ' ').title()} | You entered: {education_level}<br><br>
                    📋 Check the "View Results" tab for detailed analysis.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-alert">
                    🎉 Evaluation completed successfully! Check the "View Results" tab for detailed analysis.
                </div>
                """, unsafe_allow_html=True)

            st.balloons()

        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.exception(e)


def safe_get(dictionary, key, default=0):
    """Safely get value from dictionary with fallback"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, default)
    return default


def show_evaluation_results():
    """Step 3: Show detailed evaluation results"""
    if not st.session_state.evaluation_results:
        st.warning("⚠️ No evaluation results available. Please complete Steps 1 and 2 first.")
        return

    results = st.session_state.evaluation_results
    evaluation_data = results['evaluation_data']

    st.markdown("""
    <div class="main-card">
        <h2>📊 Step 3: Hybrid Evaluation Results</h2>
        <p>Comprehensive AI analysis with intelligent data fusion</p>
    </div>
    """, unsafe_allow_html=True)

    final_scores = evaluation_data.get('final_scores', {})
    score = safe_get(final_scores, 'overall_score', 30)
    verdict = final_scores.get('verdict', 'Low')
    llm_powered = final_scores.get('llm_powered', False)
    hybrid_data = final_scores.get('hybrid_data', {})

    # Color coding
    if score >= 80:
        score_color = "#4CAF50"
        verdict_emoji = "🟢"
    elif score >= 50:
        score_color = "#FF9800"
        verdict_emoji = "🟡"
    else:
        score_color = "#F44336"
        verdict_emoji = "🔴"

    # Hero score section
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        hybrid_badge = "🤖 AI + 🧠 Hybrid" if hybrid_data else "🔧 Standard"

        st.markdown(f"""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, {score_color}22, {score_color}44); border-radius: 20px; margin: 2rem 0;">
            <h1 style="font-size: 5rem; color: {score_color}; margin: 0;">{score}</h1>
            <h3 style="color: {score_color}; margin: 0.5rem 0;">/ 100</h3>
            <h2 style="color: {score_color}; margin: 1rem 0;">{verdict_emoji} {verdict} Match</h2>
            <p style="color: #666; margin: 0; font-size: 1.1rem;">Compatibility Score</p>
            <p style="color: #888; margin-top: 1rem;">Candidate: {results['candidate_name']}</p>
            <p style="color: #999; font-size: 0.9rem;">Analysis: {'🤖 AI-Powered' if llm_powered else '🔧 Rule-based'} | {hybrid_badge}</p>
        </div>
        """, unsafe_allow_html=True)

    # Detailed analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Score Breakdown", "🔄 Hybrid Data", "💡 AI Feedback"])

    with tab1:
        st.markdown("### 📊 Detailed Score Analysis")

        col1, col2 = st.columns(2)

        with col1:
            component_scores = final_scores.get('component_scores', {})

            st.markdown("#### Component Scores")

            skill_score = (safe_get(component_scores, 'skill_match') or
                           safe_get(component_scores, 'keyword_score') or
                           safe_get(component_scores, 'keyword_density') or 50)

            semantic_score = (safe_get(component_scores, 'domain_relevance') or
                              safe_get(component_scores, 'semantic_score') or 50)

            experience_score = (safe_get(component_scores, 'experience_match') or
                                safe_get(component_scores, 'experience_bonus') or 0)

            education_score = (safe_get(component_scores, 'education_match') or
                               safe_get(component_scores, 'education_bonus') or 0)

            st.metric("🔤 Skill Matching", f"{skill_score:.1f}%")
            st.progress(min(100, max(0, skill_score)) / 100)

            st.metric("🧠 Domain Relevance", f"{semantic_score:.1f}%")
            st.progress(min(100, max(0, semantic_score)) / 100)

            st.metric("💼 Experience Match", f"{experience_score:.1f}%")
            st.progress(min(100, max(0, experience_score)) / 100)

            st.metric("🎓 Education Match", f"{education_score:.1f}%")
            st.progress(min(100, max(0, education_score)) / 100)

        with col2:
            st.markdown("#### Evaluation Details")

            processing_time = evaluation_data.get('processing_time_seconds', 0)
            st.metric("⏱️ Processing Time", f"{processing_time:.2f} seconds")

            method_display = "🤖 AI Analysis (Gemini)" if llm_powered else "🔧 Enhanced Rule-based"
            st.markdown(f"**🔄 Analysis Method:** {method_display}")

            if hybrid_data:
                st.markdown("**🔄 Data Method:** 🧠 Hybrid Intelligence")
            else:
                st.markdown("**🔄 Data Method:** 📄 Standard Extraction")

            detailed_results = evaluation_data.get('detailed_results', {})
            vector_sim = detailed_results.get('vector_similarity', 0)
            st.metric("🔗 Vector Similarity", f"{vector_sim:.3f}")

    with tab2:
        st.markdown("### 🔄 Hybrid Data Analysis")

        if hybrid_data:
            st.markdown("**🧠 Intelligent Data Resolution Results:**")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💼 Experience Analysis")

                final_exp = hybrid_data.get('experience_years', 0)
                exp_source = hybrid_data.get('experience_source', 'unknown')
                extracted_exp = hybrid_data.get('extracted_experience', 0)
                manual_exp = hybrid_data.get('manual_experience', 0)

                st.info(f"**Final Decision:** {final_exp} years ({exp_source})")

                st.markdown("**Data Sources:**")
                st.markdown(f"• 📄 **From Resume:** {extracted_exp} years")
                st.markdown(f"• ✏️ **Manual Input:** {manual_exp} years")

                if "auto-extracted" in exp_source and "verified" in exp_source:
                    st.success("✅ Resume data verified by manual input")
                elif "manual" in exp_source and "corrected" in exp_source:
                    st.warning("⚠️ Manual input used to correct resume extraction")
                elif "auto-extracted" in exp_source:
                    st.info("📄 Used resume content (more reliable)")
                elif "manual" in exp_source:
                    st.info("✏️ Used manual input (resume unclear)")

            with col2:
                st.markdown("#### 🎓 Education Analysis")

                final_edu = hybrid_data.get('education_level', 'unknown')
                edu_source = hybrid_data.get('education_source', 'unknown')
                extracted_edu = hybrid_data.get('extracted_education', 'not_specified')
                manual_edu = hybrid_data.get('manual_education', 'not_specified')

                st.info(f"**Final Decision:** {final_edu.replace('_', ' ').title()} ({edu_source})")

                st.markdown("**Data Sources:**")
                st.markdown(f"• 📄 **From Resume:** {extracted_edu.replace('_', ' ').title()}")
                st.markdown(f"• ✏️ **Manual Input:** {manual_edu.title()}")

                if "auto-extracted" in edu_source and "verified" in edu_source:
                    st.success("✅ Resume data verified by manual input")
                elif "manual" in edu_source and "higher" in edu_source:
                    st.warning("📈 Manual input shows higher qualification")
                elif "auto-extracted" in edu_source:
                    st.info("📄 Used resume content (more reliable)")
                elif "manual" in edu_source:
                    st.info("✏️ Used manual input (resume unclear)")

        else:
            st.info("Hybrid data analysis not available - using standard extraction only.")

    with tab3:
        st.markdown("### 💡 AI-Generated Feedback & Recommendations")

        feedback = final_scores.get('feedback', 'No detailed feedback available.')

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 2rem; border-radius: 15px; border-left: 5px solid {score_color};">
            <h4 style="color: {score_color}; margin-top: 0;">📝 Detailed Analysis</h4>
            {feedback}
        </div>
        """, unsafe_allow_html=True)

        recommendations = final_scores.get('recommendations', [])
        if recommendations:
            st.markdown("#### 🚀 Action Items")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

        missing_skills = final_scores.get('missing_skills', [])
        if missing_skills:
            st.markdown("#### 🎯 Skills to Develop")
            cols = st.columns(3)
            for i, skill in enumerate(missing_skills[:15]):
                with cols[i % 3]:
                    st.markdown(f"• **{skill.title()}**")


def load_sample_job_description():
    """Load the sample JD provided"""
    sample_jd = """1. Data Science Interns
• Internship Duration: 6 months, followed by permanent employment based on performance
• Bond: 2.6 years, including the internship period (total 30 months from joining)
• Role Overview:
  • Work on data engineering, data visualization, and data science tasks.
  • Build deep learning models using Generative AI, Computer Vision, and NLP.
  • Apply ML and deep learning algorithms effectively.
  • Work with data visualization tools like Tableau or Power BI.
  • Perform data processing using Pandas and Spark.
  • Possess strong analytical and problem-solving skills.
• Eligibility Criteria:
  • Qualification: B.Tech, BE
  • Batch Eligibility: 2023 and earlier pass-outs
  • Job Types: Full-time, Internship, Fresher, Permanent
  • Stipend: ₹5,000 per month
  • Schedule: Day shift, Monday to Friday
  • Location: Pune (Onsite)

2. Data Engineer Intern
• Location: Pune (Onsite)
• Internship Duration: 6 months
• Bond: 24 months post internship if converted to job
• Requirements:
  • Formal training on Python & Spark
  • No prior experience required; 2022 or earlier graduates preferred
• Job Responsibilities:
  • Build scalable streaming data pipelines
  • Write complex SQL queries to transform source data
  • Write stable, enterprise-grade code
  • Deploy data pipelines with DevOps team
  • Build automated job scheduling and monitoring scripts
• Skills:
  • Exceptional programming skills in Python, Spark, Kafka, Pyspark, and C++
  • Strong SQL and complex query writing skills
  • Knowledge of Pandas, Numpy, and Databricks(advantageous)
  • Familiarity with Exploratory Data Analysis and data pre-processing
• Eligibility Criteria:
  • Qualification: B.Tech, BE
  • Batch Eligibility: 2023 and earlier pass-outs
  • Job Types: Full-time, Internship, Fresher, Permanent
  • Stipend: ₹5,000 per month
  • Schedule: Day shift, Monday to Friday"""

    st.session_state.job_description = sample_jd
    st.success("✅ Sample job description loaded! You can modify it or proceed to Step 2.")
    st.rerun()


def reset_application():
    """Reset all session state"""
    st.session_state.job_description = ""
    st.session_state.evaluation_results = None
    st.session_state.current_step = 1
    st.success("🔄 Application reset! Start fresh from Step 1.")
    st.rerun()


if __name__ == "__main__":
    main()