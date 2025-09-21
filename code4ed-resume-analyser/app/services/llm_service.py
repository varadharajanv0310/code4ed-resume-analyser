"""
Enhanced LLM service using Google Gemini API with smart quota management
"""
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
import time
import json
import logging
import re
import os
import streamlit as st
from typing import Dict, List, Optional, Tuple
from app.config import Config


class EnhancedLLMService:
    """Enhanced LLM service with better quota management and prompts"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        self.request_count = 0
        self.daily_limit = 45  # Conservative limit to avoid quota issues
        self.min_request_interval = 3  # 3 seconds between requests

        # Load API key (first try Config/env, then Streamlit secrets)
        self.api_key = (
            getattr(Config, "GOOGLE_API_KEY", None)
            or os.getenv("GOOGLE_API_KEY")
            or st.secrets.get("GOOGLE_API_KEY")
        )

        # Configure Gemini API if key is available
        if self.api_key:
            genai.configure(api_key=self.api_key)

            # Initialize with optimized settings
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.3,  # Balanced creativity and consistency
                convert_system_message_to_human=True,
                request_timeout=30,  # 30 second timeout
                max_retries=1  # Don't retry on quota errors
            )

            self.logger.info("Enhanced Gemini LLM initialized successfully")
        else:
            self.logger.error("Google API key not found in configuration or secrets")
            self.llm = None

    def _check_rate_limit(self) -> bool:
        """Check if we can make a request without hitting rate limits"""
        current_time = time.time()

        # Check minimum interval between requests
        if current_time - self.last_request_time < self.min_request_interval:
            return False

        # Simple daily counter (resets on restart)
        if self.request_count >= self.daily_limit:
            return False

        return True

    def _make_llm_request(self, messages: List) -> Optional[str]:
        """Make LLM request with rate limiting and error handling"""
        if not self.llm:
            return None

        if not self._check_rate_limit():
            self.logger.warning("Rate limit check failed - skipping LLM request")
            return None

        try:
            # Wait for minimum interval
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)

            response = self.llm.invoke(messages)

            # Update rate limiting counters
            self.last_request_time = time.time()
            self.request_count += 1

            self.logger.info(f"LLM request successful. Count: {self.request_count}/{self.daily_limit}")
            return response.content

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                self.logger.warning(f"LLM quota exceeded: {error_str}")
                return None
            else:
                self.logger.error(f"LLM request failed: {error_str}")
                return None

    def generate_comprehensive_evaluation(self, resume_text: str, jd_text: str) -> Dict[str, any]:
        """Single comprehensive LLM evaluation call to minimize API usage"""

        # Optimized prompt that gets everything in one call
        prompt = f"""You are an expert HR analyst. Analyze this resume against the job description comprehensively.

JOB DESCRIPTION:
{jd_text[:2000]}

RESUME:
{resume_text[:2000]}

Provide a complete evaluation in this EXACT JSON format (no additional text):
{{
    "overall_score": [0-100 integer],
    "verdict": "[High/Medium/Low]",
    "component_scores": {{
        "skill_match": [0-100],
        "experience_match": [0-100], 
        "education_match": [0-100],
        "domain_relevance": [0-100]
    }},
    "strengths": ["strength1", "strength2", "strength3"],
    "weaknesses": ["weakness1", "weakness2", "weakness3"],
    "missing_critical_skills": ["skill1", "skill2", "skill3"],
    "recommendations": ["rec1", "rec2", "rec3"],
    "detailed_feedback": "Comprehensive feedback paragraph with specific actionable advice for the candidate.",
    "hiring_recommendation": "Detailed recommendation for hiring managers"
}}

Be precise, specific, and actionable in your analysis."""

        messages = [
            SystemMessage(
                content="You are an expert HR analyst specializing in resume evaluation. Always respond with valid JSON only."),
            HumanMessage(content=prompt)
        ]

        response = self._make_llm_request(messages)

        if response:
            try:
                # Clean response (remove any markdown formatting)
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                # Parse JSON response
                evaluation = json.loads(cleaned_response)

                # Validate required fields
                required_fields = ['overall_score', 'verdict', 'detailed_feedback']
                if all(field in evaluation for field in required_fields):
                    # Ensure score is within bounds
                    evaluation['overall_score'] = max(20, min(100, evaluation['overall_score']))
                    return evaluation
                else:
                    self.logger.warning("LLM response missing required fields")
                    return None

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM JSON response: {e}")
                # Try to extract score from text if JSON parsing fails
                return self._extract_score_from_text(response)

        return None

    def _extract_score_from_text(self, text: str) -> Dict[str, any]:
        """Extract score and basic info from malformed LLM response"""
        # Try to extract score
        score_match = re.search(r'"overall_score":\s*(\d+)', text)
        score = int(score_match.group(1)) if score_match else 50

        # Try to extract verdict
        verdict_match = re.search(r'"verdict":\s*"(\w+)"', text)
        verdict = verdict_match.group(1) if verdict_match else (
            "High" if score >= 75 else "Medium" if score >= 50 else "Low")

        return {
            "overall_score": max(20, min(100, score)),
            "verdict": verdict,
            "detailed_feedback": text[:500] + "..." if len(text) > 500 else text,
            "component_scores": {
                "skill_match": score,
                "experience_match": score,
                "education_match": score,
                "domain_relevance": score
            },
            "strengths": ["Analysis available in detailed feedback"],
            "weaknesses": ["See detailed feedback for improvement areas"],
            "missing_critical_skills": ["Refer to detailed feedback"],
            "recommendations": ["Check detailed feedback for recommendations"]
        }

    def generate_semantic_analysis(self, resume_text: str, jd_text: str) -> Dict[str, any]:
        """Generate semantic analysis (legacy method for compatibility)"""
        result = self.generate_comprehensive_evaluation(resume_text, jd_text)
        if result:
            return {
                "semantic_score": result['overall_score'],
                "strengths": result.get('strengths', []),
                "weaknesses": result.get('weaknesses', []),
                "recommendation": "interview" if result['overall_score'] >= 70 else "consider" if result[
                                                                                                      'overall_score'] >= 50 else "reject"
            }
        return {"semantic_score": 50, "error": "LLM not available"}

    def generate_feedback(self, resume_text: str, jd_text: str, relevance_score: int) -> str:
        """Generate personalized feedback (legacy method for compatibility)"""
        result = self.generate_comprehensive_evaluation(resume_text, jd_text)
        if result:
            return result.get('detailed_feedback', 'No detailed feedback available.')
        return f"Unable to generate detailed feedback. LLM service unavailable. Score: {relevance_score}/100"

    def test_connection(self) -> Dict[str, any]:
        """Test LLM connection and return status"""
        if not self.llm:
            return {"status": "failed", "reason": "LLM not initialized"}

        if not self._check_rate_limit():
            return {"status": "quota_exceeded",
                    "reason": f"Daily limit reached ({self.request_count}/{self.daily_limit})"}

        test_response = self._make_llm_request([
            HumanMessage(content="Respond with exactly: test successful")
        ])

        if test_response:
            return {
                "status": "success",
                "remaining_quota": self.daily_limit - self.request_count,
                "response": test_response
            }
        else:
            return {"status": "failed", "reason": "API quota exceeded or connection failed"}


# Global enhanced LLM service instance
enhanced_llm_service = EnhancedLLMService()

# Legacy compatibility
llm_service = enhanced_llm_service
