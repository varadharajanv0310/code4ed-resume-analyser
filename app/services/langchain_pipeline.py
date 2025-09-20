"""
LangChain pipeline for orchestrating resume evaluation workflow
"""
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

from app.services.llm_service import llm_service
from app.services.text_processor import text_processor
from app.services.keyword_matcher import keyword_matcher
from app.services.vector_store import vector_store
from app.services.scorer import relevance_scorer


class ResumeEvaluationPipeline:
    """LangChain-based pipeline for comprehensive resume evaluation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory = ConversationBufferMemory()

        # Initialize LangChain components
        if llm_service.llm:
            self.evaluation_chain = self._create_evaluation_chain()
            self.feedback_chain = self._create_feedback_chain()
        else:
            self.logger.warning("LLM not available - pipeline will use fallback methods")
            self.evaluation_chain = None
            self.feedback_chain = None

    def _create_evaluation_chain(self) -> Optional[LLMChain]:
        """Create LangChain evaluation chain"""
        try:
            prompt = PromptTemplate(
                input_variables=["resume_text", "jd_text", "keyword_score", "vector_score"],
                template="""
You are an expert resume evaluator. Analyze this resume against the job requirements.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

PRELIMINARY SCORES:
- Keyword Match: {keyword_score}%
- Vector Similarity: {vector_score}%

Provide your evaluation in this JSON format:
{{
    "semantic_relevance": [0-100],
    "skill_gaps": ["list of missing skills"],
    "strengths": ["key strengths"],
    "experience_fit": "poor/fair/good/excellent",
    "overall_assessment": "brief summary",
    "confidence_score": [0-100]
}}

Focus on semantic understanding beyond just keyword matching.
"""
            )

            return LLMChain(
                llm=llm_service.llm,
                prompt=prompt,
                memory=self.memory,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"Failed to create evaluation chain: {str(e)}")
            return None

    def _create_feedback_chain(self) -> Optional[LLMChain]:
        """Create LangChain feedback generation chain"""
        try:
            prompt = PromptTemplate(
                input_variables=["resume_text", "jd_text", "evaluation_results", "final_score"],
                template="""
As a career coach, provide constructive feedback to help improve this resume.

JOB REQUIREMENTS:
{jd_text}

CURRENT RESUME:
{resume_text}

EVALUATION RESULTS:
{evaluation_results}

FINAL SCORE: {final_score}/100

Provide actionable feedback in this structure:
1. STRENGTHS (what's working well)
2. IMPROVEMENT AREAS (specific changes needed)
3. SKILL DEVELOPMENT (skills to learn/improve)
4. FORMATTING SUGGESTIONS
5. ACTION ITEMS (concrete next steps)

Be encouraging but specific about improvements needed.
"""
            )

            return LLMChain(
                llm=llm_service.llm,
                prompt=prompt,
                memory=self.memory,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"Failed to create feedback chain: {str(e)}")
            return None

    def process_resume_evaluation(self, resume_text: str, jd_text: str,
                                  resume_id: str = None, job_id: str = None) -> Dict[str, Any]:
        """Complete resume evaluation pipeline"""
        start_time = datetime.now()

        try:
            # Step 1: Text Processing
            self.logger.info(f"Processing resume {resume_id} for job {job_id}")
            resume_processed = text_processor.process_text(resume_text)
            jd_processed = text_processor.process_text(jd_text)

            # Step 2: Keyword Matching
            keyword_results = keyword_matcher.comprehensive_keyword_matching(
                resume_text, jd_text,
                resume_processed['skills'], jd_processed['skills']
            )

            # Step 3: Vector Similarity (if available)
            vector_similarity = 0.0
            if vector_store.client and resume_id and job_id:
                vector_similarity = vector_store.calculate_resume_jd_similarity(resume_id, job_id)

            # Step 4: LLM Semantic Analysis
            semantic_results = {}
            if self.evaluation_chain:
                try:
                    llm_response = self.evaluation_chain.run({
                        "resume_text": resume_text,
                        "jd_text": jd_text,
                        "keyword_score": keyword_results['combined_percentage'],
                        "vector_score": vector_similarity * 100
                    })

                    # Try to parse JSON response
                    try:
                        semantic_results = json.loads(llm_response)
                    except json.JSONDecodeError:
                        semantic_results = {"raw_response": llm_response}

                except Exception as e:
                    self.logger.error(f"LLM evaluation failed: {str(e)}")
                    semantic_results = {"error": str(e)}

            # Step 5: Comprehensive Scoring
            final_scores = relevance_scorer.calculate_comprehensive_score(
                keyword_results=keyword_results,
                semantic_results=semantic_results,
                vector_similarity=vector_similarity,
                resume_experience=resume_processed['experience_years'],
                resume_education=resume_processed['education_level']
            )

            # Step 6: Generate Feedback
            feedback = ""
            if self.feedback_chain:
                try:
                    feedback = self.feedback_chain.run({
                        "resume_text": resume_text,
                        "jd_text": jd_text,
                        "evaluation_results": json.dumps(semantic_results, indent=2),
                        "final_score": final_scores['overall_score']
                    })
                except Exception as e:
                    self.logger.error(f"Feedback generation failed: {str(e)}")
                    feedback = f"Feedback generation failed: {str(e)}"

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Compile final results
            evaluation_results = {
                'resume_id': resume_id,
                'job_id': job_id,
                'processing_time_seconds': processing_time,
                'processed_resume': resume_processed,
                'processed_jd': jd_processed,
                'keyword_analysis': keyword_results,
                'semantic_analysis': semantic_results,
                'vector_similarity': vector_similarity,
                'final_scores': final_scores,
                'feedback': feedback,
                'pipeline_status': 'completed',
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"Evaluation completed for resume {resume_id}: {final_scores['overall_score']}/100")
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Pipeline evaluation failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'resume_id': resume_id,
                'job_id': job_id,
                'processing_time_seconds': processing_time,
                'pipeline_status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'final_scores': {
                    'overall_score': 30,
                    'verdict': 'Low',
                    'error': 'Pipeline processing failed'
                }
            }

    def batch_evaluate_resumes(self, resume_job_pairs: List[Dict]) -> List[Dict]:
        """Evaluate multiple resume-job pairs"""
        results = []

        for i, pair in enumerate(resume_job_pairs):
            self.logger.info(f"Processing batch item {i + 1}/{len(resume_job_pairs)}")

            result = self.process_resume_evaluation(
                resume_text=pair.get('resume_text', ''),
                jd_text=pair.get('jd_text', ''),
                resume_id=pair.get('resume_id'),
                job_id=pair.get('job_id')
            )

            results.append(result)

            # Add batch metadata
            result['batch_index'] = i
            result['batch_total'] = len(resume_job_pairs)

        # Sort results by score (highest first)
        results.sort(key=lambda x: x.get('final_scores', {}).get('overall_score', 0), reverse=True)

        return results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        return {
            'components_status': {
                'text_processor': text_processor is not None,
                'keyword_matcher': keyword_matcher is not None,
                'vector_store': vector_store.client is not None,
                'llm_service': llm_service.llm is not None,
                'scorer': relevance_scorer is not None
            },
            'langchain_chains': {
                'evaluation_chain': self.evaluation_chain is not None,
                'feedback_chain': self.feedback_chain is not None
            },
            'memory_messages': len(self.memory.chat_memory.messages) if self.memory else 0
        }

    def clear_pipeline_memory(self):
        """Clear LangChain conversation memory"""
        if self.memory:
            self.memory.clear()
            self.logger.info("Pipeline memory cleared")


# Global pipeline instance
resume_pipeline = ResumeEvaluationPipeline()