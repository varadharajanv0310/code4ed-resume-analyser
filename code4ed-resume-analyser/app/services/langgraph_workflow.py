"""
Complete LangGraph stateful workflow for structured resume-JD analysis
With hybrid manual/extracted data integration
"""
from typing import Dict, List, Any, TypedDict, Annotated
from datetime import datetime
import logging


class WorkflowState(TypedDict):
    """State structure for the resume evaluation workflow"""
    resume_text: str
    jd_text: str
    resume_id: str
    job_id: str
    current_step: str
    processed_resume: Dict[str, Any]
    processed_jd: Dict[str, Any]
    keyword_results: Dict[str, Any]
    semantic_results: Dict[str, Any]
    vector_similarity: float
    final_scores: Dict[str, Any]
    feedback: str
    errors: List[str]
    processing_start: datetime
    step_history: List[str]
    manual_experience: int
    manual_education: str


class ResumeWorkflowGraph:
    """Complete LangGraph workflow for resume evaluation with hybrid data support"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Import services
        from app.services.text_processor import text_processor
        from app.services.keyword_matcher import keyword_matcher
        from app.services.vector_store import vector_store
        from app.services.llm_service import enhanced_llm_service
        from app.services.scorer import llm_integrated_scorer

        self.text_processor = text_processor
        self.keyword_matcher = keyword_matcher
        self.vector_store = vector_store
        self.llm_service = enhanced_llm_service
        self.scorer = llm_integrated_scorer

        # Define workflow steps
        self.workflow_steps = [
            'initialize',
            'process_texts',
            'keyword_matching',
            'vector_analysis',
            'semantic_analysis',
            'scoring',
            'feedback_generation',
            'finalize'
        ]

    def initialize_state(self, resume_text: str, jd_text: str,
                         resume_id: str = None, job_id: str = None) -> WorkflowState:
        """Initialize workflow state"""
        return WorkflowState(
            resume_text=resume_text,
            jd_text=jd_text,
            resume_id=resume_id or f"resume_{datetime.now().timestamp()}",
            job_id=job_id or f"job_{datetime.now().timestamp()}",
            current_step='initialize',
            processed_resume={},
            processed_jd={},
            keyword_results={},
            semantic_results={},
            vector_similarity=0.0,
            final_scores={},
            feedback="",
            errors=[],
            processing_start=datetime.now(),
            step_history=[],
            manual_experience=None,
            manual_education=None
        )

    def step_initialize(self, state: WorkflowState) -> WorkflowState:
        """Initialize the workflow"""
        try:
            state['current_step'] = 'initialize'
            state['step_history'].append('initialize')
            self.logger.info(f"Initializing hybrid workflow for resume {state['resume_id']}")

            # Validate inputs
            if not state['resume_text'].strip():
                state['errors'].append("Resume text is empty")
            if not state['jd_text'].strip():
                state['errors'].append("Job description text is empty")

            # Check LLM service status
            llm_status = self.llm_service.test_connection()
            self.logger.info(f"LLM service status: {llm_status}")

            # Log manual data if provided
            if state.get('manual_experience') is not None:
                self.logger.info(f"Manual experience provided: {state['manual_experience']} years")
            if state.get('manual_education'):
                self.logger.info(f"Manual education provided: {state['manual_education']}")

            state['current_step'] = 'process_texts'
            return state

        except Exception as e:
            state['errors'].append(f"Initialize step failed: {str(e)}")
            self.logger.error(f"Initialize step failed: {str(e)}")
            return state

    def step_process_texts(self, state: WorkflowState) -> WorkflowState:
        """Process resume and job description texts"""
        try:
            state['current_step'] = 'process_texts'
            state['step_history'].append('process_texts')

            if state['errors']:  # Skip if previous errors
                state['current_step'] = 'finalize'
                return state

            # Process resume text
            state['processed_resume'] = self.text_processor.process_text(state['resume_text'])

            # Process job description text
            state['processed_jd'] = self.text_processor.process_text(state['jd_text'])

            self.logger.info(f"Text processing completed for {state['resume_id']}")
            state['current_step'] = 'keyword_matching'
            return state

        except Exception as e:
            state['errors'].append(f"Text processing failed: {str(e)}")
            self.logger.error(f"Text processing failed: {str(e)}")
            state['current_step'] = 'semantic_analysis'  # Skip to LLM analysis
            return state

    def step_keyword_matching(self, state: WorkflowState) -> WorkflowState:
        """Perform comprehensive keyword matching"""
        try:
            state['current_step'] = 'keyword_matching'
            state['step_history'].append('keyword_matching')

            if state['errors']:
                state['current_step'] = 'semantic_analysis'
                return state

            # Perform keyword matching
            state['keyword_results'] = self.keyword_matcher.comprehensive_keyword_matching(
                state['resume_text'],
                state['jd_text'],
                state['processed_resume'].get('skills', set()),
                state['processed_jd'].get('skills', set())
            )

            match_percentage = state['keyword_results'].get('combined_percentage', 0)
            self.logger.info(f"Keyword matching completed: {match_percentage}%")
            state['current_step'] = 'vector_analysis'
            return state

        except Exception as e:
            state['errors'].append(f"Keyword matching failed: {str(e)}")
            self.logger.error(f"Keyword matching failed: {str(e)}")
            state['keyword_results'] = {'combined_percentage': 50, 'error': str(e)}
            state['current_step'] = 'vector_analysis'
            return state

    def step_vector_analysis(self, state: WorkflowState) -> WorkflowState:
        """Perform vector similarity analysis"""
        try:
            state['current_step'] = 'vector_analysis'
            state['step_history'].append('vector_analysis')

            if self.vector_store.client:
                # Add documents to vector store if not already present
                try:
                    self.vector_store.add_resume(
                        state['resume_id'],
                        state['resume_text'],
                        metadata={'processed': True}
                    )
                    self.vector_store.add_job_description(
                        state['job_id'],
                        state['jd_text'],
                        metadata={'processed': True}
                    )

                    # Calculate similarity
                    state['vector_similarity'] = self.vector_store.calculate_resume_jd_similarity(
                        state['resume_id'], state['job_id']
                    )

                    self.logger.info(f"Vector analysis completed: {state['vector_similarity']:.4f}")
                except Exception as vector_error:
                    self.logger.warning(f"Vector store operation failed: {vector_error}")
                    state['vector_similarity'] = 0.0
            else:
                self.logger.warning("Vector store not available")
                state['vector_similarity'] = 0.0

            state['current_step'] = 'semantic_analysis'
            return state

        except Exception as e:
            state['errors'].append(f"Vector analysis failed: {str(e)}")
            self.logger.error(f"Vector analysis failed: {str(e)}")
            state['vector_similarity'] = 0.0
            state['current_step'] = 'semantic_analysis'
            return state

    def step_semantic_analysis(self, state: WorkflowState) -> WorkflowState:
        """Enhanced semantic analysis with comprehensive LLM evaluation"""
        try:
            state['current_step'] = 'semantic_analysis'
            state['step_history'].append('semantic_analysis')

            # Try comprehensive LLM evaluation
            self.logger.info("Attempting LLM-powered evaluation...")

            llm_evaluation = self.llm_service.generate_comprehensive_evaluation(
                state['resume_text'],
                state['jd_text']
            )

            if llm_evaluation:
                state['semantic_results'] = llm_evaluation
                score = llm_evaluation.get('overall_score', 'N/A')
                self.logger.info(f"LLM evaluation completed successfully: {score}/100")
            else:
                # LLM unavailable - mark for fallback in scoring step
                state['semantic_results'] = {"llm_unavailable": True}
                self.logger.warning("LLM evaluation unavailable - will use enhanced fallback")

            state['current_step'] = 'scoring'
            return state

        except Exception as e:
            state['errors'].append(f"Semantic analysis failed: {str(e)}")
            self.logger.error(f"Semantic analysis failed: {str(e)}")
            state['semantic_results'] = {"error": str(e)}
            state['current_step'] = 'scoring'
            return state

    def step_scoring_hybrid(self, state: WorkflowState) -> WorkflowState:
        """Enhanced scoring that integrates LLM results with hybrid manual/extracted data"""
        try:
            state['current_step'] = 'scoring'
            state['step_history'].append('scoring')

            # Process LLM results with hybrid approach
            llm_result = state['semantic_results'] if not state['semantic_results'].get('error') else None

            state['final_scores'] = self.scorer.process_llm_evaluation_with_manual_data(
                llm_result,
                state['resume_text'],
                state['jd_text'],
                state.get('manual_experience'),
                state.get('manual_education')
            )

            final_score = state['final_scores']['overall_score']
            verdict = state['final_scores']['verdict']
            llm_powered = state['final_scores'].get('llm_powered', False)
            source = state['final_scores'].get('source', 'Unknown')

            # Log hybrid data usage
            hybrid_data = state['final_scores'].get('hybrid_data', {})
            exp_info = f"{hybrid_data.get('experience_years', 0)} years ({hybrid_data.get('experience_source', 'unknown')})"
            edu_info = f"{hybrid_data.get('education_level', 'unknown')} ({hybrid_data.get('education_source', 'unknown')})"

            self.logger.info(f"Hybrid scoring completed: {final_score}/100 ({verdict}) - Source: {source}")
            self.logger.info(f"Experience: {exp_info}, Education: {edu_info}")

            state['current_step'] = 'feedback_generation'
            return state

        except Exception as e:
            state['errors'].append(f"Hybrid scoring failed: {str(e)}")
            self.logger.error(f"Hybrid scoring failed: {str(e)}")

            # Emergency fallback
            state['final_scores'] = {
                "overall_score": 40,
                "verdict": "Low",
                "error": str(e),
                "feedback": f"Scoring system error: {str(e)}",
                "llm_powered": False,
                "source": "Emergency Fallback"
            }
            state['current_step'] = 'feedback_generation'
            return state

    def step_feedback_generation(self, state: WorkflowState) -> WorkflowState:
        """Enhanced feedback generation with multiple sources"""
        try:
            state['current_step'] = 'feedback_generation'
            state['step_history'].append('feedback_generation')

            # Check if we already have feedback from the scoring step
            if state['final_scores'].get('feedback'):
                state['feedback'] = state['final_scores']['feedback']
                self.logger.info("Using feedback from hybrid scoring system")
            else:
                # Generate basic feedback as fallback
                state['feedback'] = self.generate_basic_feedback(state)
                self.logger.info("Generated basic feedback")

            state['current_step'] = 'finalize'
            return state

        except Exception as e:
            state['errors'].append(f"Feedback generation failed: {str(e)}")
            self.logger.error(f"Feedback generation failed: {str(e)}")
            state['feedback'] = f"Feedback generation failed: {str(e)}"
            state['current_step'] = 'finalize'
            return state

    def step_finalize(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow with comprehensive results"""
        state['current_step'] = 'finalize'
        state['step_history'].append('finalize')

        processing_time = (datetime.now() - state['processing_start']).total_seconds()

        # Log final results
        final_score = state['final_scores'].get('overall_score', 0)
        verdict = state['final_scores'].get('verdict', 'Unknown')
        llm_powered = state['final_scores'].get('llm_powered', False)

        # Log hybrid data summary
        hybrid_data = state['final_scores'].get('hybrid_data', {})
        if hybrid_data:
            exp_summary = f"{hybrid_data.get('experience_years', 0)} years ({hybrid_data.get('experience_source', 'unknown')})"
            edu_summary = f"{hybrid_data.get('education_level', 'unknown')} ({hybrid_data.get('education_source', 'unknown')})"
            self.logger.info(f"Final hybrid data: Experience: {exp_summary}, Education: {edu_summary}")

        self.logger.info(f"Workflow completed for {state['resume_id']} in {processing_time:.2f}s")
        self.logger.info(f"Final results: {final_score}/100 ({verdict}) - LLM: {llm_powered}")

        if state['errors']:
            self.logger.warning(f"Workflow completed with {len(state['errors'])} errors: {state['errors']}")

        return state

    def generate_basic_feedback(self, state: WorkflowState) -> str:
        """Generate basic feedback when other methods fail"""
        try:
            score = state['final_scores'].get('overall_score', 50)
            verdict = state['final_scores'].get('verdict', 'Medium')
            missing_skills = state['final_scores'].get('missing_skills', [])
            hybrid_data = state['final_scores'].get('hybrid_data', {})

            feedback_parts = []

            # Score-based opening
            if score >= 75:
                feedback_parts.append(
                    "🎉 **Strong Candidate!** Your resume shows excellent alignment with the job requirements.")
            elif score >= 55:
                feedback_parts.append("👍 **Good Potential!** Your resume meets many of the job requirements.")
            elif score >= 35:
                feedback_parts.append(
                    "⚠️ **Moderate Fit.** Your resume has some relevant qualifications but needs improvement.")
            else:
                feedback_parts.append(
                    "📈 **Significant Development Needed.** Your resume requires substantial improvement for this role.")

            feedback_parts.append(f"\n**Evaluation Summary:**")
            feedback_parts.append(f"• Overall Score: {score}/100")
            feedback_parts.append(f"• Match Level: {verdict}")

            # Add hybrid data info if available
            if hybrid_data:
                exp_years = hybrid_data.get('experience_years', 0)
                exp_source = hybrid_data.get('experience_source', 'unknown')
                edu_level = hybrid_data.get('education_level', 'unknown')
                edu_source = hybrid_data.get('education_source', 'unknown')

                feedback_parts.append(f"• Experience: {exp_years} years ({exp_source})")
                feedback_parts.append(f"• Education: {edu_level.replace('_', ' ').title()} ({edu_source})")

            # Add processing info
            llm_powered = state['final_scores'].get('llm_powered', False)
            source = state['final_scores'].get('source', 'Basic Analysis')
            feedback_parts.append(f"• Analysis Method: {source}")

            if missing_skills:
                feedback_parts.append(f"\n**Key Skills to Develop:**")
                for skill in missing_skills[:5]:
                    feedback_parts.append(f"• {skill}")

            # Basic recommendations
            feedback_parts.append(f"\n**Recommendations:**")
            if score >= 70:
                feedback_parts.append("• You're well-qualified - apply with confidence")
                feedback_parts.append("• Highlight your strongest qualifications")
            elif score >= 45:
                feedback_parts.append("• Address the identified skill gaps")
                feedback_parts.append("• Consider additional training or projects")
            else:
                feedback_parts.append("• Focus on developing core required skills")
                feedback_parts.append("• Build relevant experience through projects")

            return "\n".join(feedback_parts)

        except Exception as e:
            return f"Basic feedback generation failed: {str(e)}"

    def execute_workflow(self, resume_text: str, jd_text: str,
                         resume_id: str = None, job_id: str = None,
                         manual_experience: int = None, manual_education: str = None) -> Dict[str, Any]:
        """Execute the complete hybrid workflow"""

        # Initialize state
        state = self.initialize_state(resume_text, jd_text, resume_id, job_id)

        # Store manual input data in state for later use
        state['manual_experience'] = manual_experience
        state['manual_education'] = manual_education

        # Execute workflow steps
        step_functions = {
            'initialize': self.step_initialize,
            'process_texts': self.step_process_texts,
            'keyword_matching': self.step_keyword_matching,
            'vector_analysis': self.step_vector_analysis,
            'semantic_analysis': self.step_semantic_analysis,
            'scoring': self.step_scoring_hybrid,  # Use hybrid version
            'feedback_generation': self.step_feedback_generation,
            'finalize': self.step_finalize
        }

        max_iterations = 10  # Prevent infinite loops
        iterations = 0

        while state['current_step'] != 'finalize' and iterations < max_iterations:
            step_function = step_functions.get(state['current_step'])
            if step_function:
                state = step_function(state)
            else:
                state['errors'].append(f"Unknown step: {state['current_step']}")
                break
            iterations += 1

        # Final finalization if not reached
        if state['current_step'] != 'finalize':
            state = self.step_finalize(state)

        # Calculate final processing time
        processing_time = (datetime.now() - state['processing_start']).total_seconds()

        # Return comprehensive results
        return {
            'resume_id': state['resume_id'],
            'job_id': state['job_id'],
            'processing_time_seconds': processing_time,
            'workflow_status': 'completed' if not state['errors'] else 'completed_with_errors',
            'errors': state['errors'],
            'step_history': state['step_history'],
            'final_scores': state['final_scores'],
            'feedback': state['feedback'],
            'detailed_results': {
                'processed_resume': state['processed_resume'],
                'keyword_analysis': state['keyword_results'],
                'semantic_analysis': state['semantic_results'],
                'vector_similarity': state['vector_similarity']
            },
            'timestamp': datetime.now().isoformat(),
            'llm_powered': state['final_scores'].get('llm_powered', False),
            'analysis_source': state['final_scores'].get('source', 'Unknown'),
            'hybrid_data': state['final_scores'].get('hybrid_data', {})
        }


# Global workflow instance
workflow_graph = ResumeWorkflowGraph()