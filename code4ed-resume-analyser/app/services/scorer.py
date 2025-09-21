"""
Complete enhanced scoring system with hybrid manual/extracted data integration
"""
from typing import Dict, List, Tuple, Set
import logging
import re
from app.config import Config


class LLMIntegratedScorer:
    """Complete scorer with hybrid manual/extracted data integration"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_score = Config.MIN_RELEVANCE_SCORE

        # Enhanced skill categories with weights for fallback analysis
        self.skill_categories = {
            'programming_languages': {
                'weight': 0.25,
                'skills': ['python', 'java', 'javascript', 'c++', 'r', 'scala', 'sql', 'matlab', 'sas', 'c#', 'php',
                           'ruby', 'go', 'rust']
            },
            'data_science_tools': {
                'weight': 0.20,
                'skills': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
                           'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda', 'scipy', 'statsmodels']
            },
            'big_data_tools': {
                'weight': 0.15,
                'skills': ['spark', 'hadoop', 'kafka', 'pyspark', 'databricks', 'hive', 'pig', 'storm', 'flink']
            },
            'databases': {
                'weight': 0.15,
                'skills': ['mysql', 'postgresql', 'mongodb', 'cassandra', 'redis', 'sqlite', 'oracle', 'sql server',
                           'nosql']
            },
            'visualization_tools': {
                'weight': 0.10,
                'skills': ['tableau', 'power bi', 'powerbi', 'd3.js', 'qlik', 'looker', 'grafana', 'bokeh']
            },
            'cloud_platforms': {
                'weight': 0.10,
                'skills': ['aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'heroku', 'digitalocean']
            },
            'ml_domains': {
                'weight': 0.05,
                'skills': ['machine learning', 'deep learning', 'nlp', 'computer vision',
                           'generative ai', 'ai', 'neural networks', 'reinforcement learning', 'mlops']
            }
        }

        # Education level mapping
        self.education_levels = {
            'phd': 5, 'doctorate': 5,
            'masters': 4, 'master': 4, 'msc': 4, 'mba': 4, 'm.tech': 4, 'ms': 4,
            'bachelors': 3, 'bachelor': 3, 'btech': 3, 'be': 3, 'bsc': 3, 'ba': 3, 'bs': 3,
            'diploma': 2, 'associate': 2,
            'high_school': 1, 'high school': 1, '12th': 1, 'secondary': 1
        }

    def process_llm_evaluation(self, llm_result: Dict, resume_text: str, jd_text: str) -> Dict[str, any]:
        """Process and enhance LLM evaluation results (legacy method)"""
        return self.process_llm_evaluation_with_manual_data(llm_result, resume_text, jd_text, None, None)

    def process_llm_evaluation_with_manual_data(self, llm_result: Dict, resume_text: str, jd_text: str,
                                                manual_experience: int = None, manual_education: str = None) -> Dict[
        str, any]:
        """Process LLM evaluation with hybrid manual/extracted data approach"""

        if not llm_result or llm_result.get('error') or llm_result.get('llm_unavailable'):
            self.logger.info("LLM unavailable - using enhanced fallback evaluation with hybrid data")
            return self.generate_fallback_evaluation_hybrid(resume_text, jd_text, manual_experience, manual_education)

        try:
            # Extract and validate LLM scores
            overall_score = max(self.min_score, min(100, llm_result.get('overall_score', 50)))
            verdict = llm_result.get('verdict', 'Medium')

            # Ensure verdict matches score
            if overall_score >= 75 and verdict not in ['High']:
                verdict = 'High'
            elif overall_score >= 50 and verdict not in ['High', 'Medium']:
                verdict = 'Medium'
            elif overall_score < 50:
                verdict = 'Low'

            # Add hybrid data analysis
            extracted_experience = self.extract_experience_years(resume_text)
            extracted_education = self.extract_education_level(resume_text)

            # Determine final values using hybrid approach
            final_experience, exp_source = self.resolve_experience_hybrid(extracted_experience, manual_experience)
            final_education, edu_source = self.resolve_education_hybrid(extracted_education, manual_education)

            self.logger.info(f"LLM evaluation processed: {overall_score}/100 ({verdict})")
            self.logger.info(
                f"Experience: {final_experience} years ({exp_source}), Education: {final_education} ({edu_source})")

            return {
                'overall_score': overall_score,
                'verdict': verdict,
                'component_scores': llm_result.get('component_scores', {}),
                'missing_skills': llm_result.get('missing_critical_skills', []),
                'feedback': llm_result.get('detailed_feedback', 'No detailed feedback available.'),
                'recommendations': llm_result.get('recommendations', []),
                'strengths': llm_result.get('strengths', []),
                'weaknesses': llm_result.get('weaknesses', []),
                'llm_powered': True,
                'source': 'LLM Analysis (Gemini)',
                'hiring_recommendation': llm_result.get('hiring_recommendation', ''),
                'hybrid_data': {
                    'experience_years': final_experience,
                    'experience_source': exp_source,
                    'education_level': final_education,
                    'education_source': edu_source,
                    'extracted_experience': extracted_experience,
                    'manual_experience': manual_experience,
                    'extracted_education': extracted_education,
                    'manual_education': manual_education
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to process LLM results: {e}")
            return self.generate_fallback_evaluation_hybrid(resume_text, jd_text, manual_experience, manual_education)

    def resolve_experience_hybrid(self, extracted_exp: int, manual_exp: int = None) -> tuple:
        """Resolve experience using hybrid approach"""

        # Case 1: Both available
        if extracted_exp > 0 and manual_exp is not None and manual_exp > 0:
            # If they're close (within 1 year), use extracted (trust the resume)
            if abs(extracted_exp - manual_exp) <= 1:
                return extracted_exp, "auto-extracted (verified)"
            # If manual is higher and reasonable, use manual (user correction)
            elif manual_exp > extracted_exp and manual_exp <= extracted_exp + 3:
                return manual_exp, "manual input (corrected)"
            # If extracted is much higher, use extracted (trust the resume)
            else:
                return extracted_exp, "auto-extracted (resume content)"

        # Case 2: Only extracted available
        elif extracted_exp > 0:
            return extracted_exp, "auto-extracted"

        # Case 3: Only manual available
        elif manual_exp is not None and manual_exp > 0:
            return manual_exp, "manual input"

        # Case 4: Neither available
        else:
            return 0, "not specified"

    def resolve_education_hybrid(self, extracted_edu: str, manual_edu: str = None) -> tuple:
        """Resolve education using hybrid approach"""

        # Education level hierarchy for comparison
        edu_hierarchy = {
            'high_school': 1, 'high school': 1, 'secondary': 1,
            'diploma': 2, 'associate': 2,
            'bachelors': 3, 'bachelor': 3, 'btech': 3, 'be': 3, 'bsc': 3, 'ba': 3,
            'masters': 4, 'master': 4, 'msc': 4, 'mba': 4, 'm.tech': 4,
            'phd': 5, 'doctorate': 5
        }

        extracted_level = edu_hierarchy.get(extracted_edu.lower(), 0)
        manual_level = edu_hierarchy.get(manual_edu.lower() if manual_edu else '', 0)

        # Case 1: Both available
        if extracted_level > 0 and manual_level > 0:
            # If they match level, use extracted
            if extracted_level == manual_level:
                return extracted_edu, "auto-extracted (verified)"
            # If manual is higher, use manual (user knows better)
            elif manual_level > extracted_level:
                return manual_edu, "manual input (higher qualification)"
            # If extracted is higher, use extracted (trust resume)
            else:
                return extracted_edu, "auto-extracted (resume content)"

        # Case 2: Only extracted available
        elif extracted_level > 0:
            return extracted_edu, "auto-extracted"

        # Case 3: Only manual available
        elif manual_level > 0:
            return manual_edu, "manual input"

        # Case 4: Neither available
        else:
            return 'not_specified', "not specified"

    def extract_detailed_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills by category from text"""
        text_lower = text.lower()
        found_skills = {}

        for category, data in self.skill_categories.items():
            found_skills[category] = []

            for skill in data['skills']:
                # Check for exact match and variations
                if skill in text_lower:
                    found_skills[category].append(skill)

                # Check for variations (e.g., "scikit learn" vs "scikit-learn")
                skill_variations = [
                    skill.replace('-', ' '),
                    skill.replace(' ', '-'),
                    skill.replace(' ', ''),
                ]

                for variation in skill_variations:
                    if variation in text_lower and skill not in found_skills[category]:
                        found_skills[category].append(skill)

        return found_skills

    def calculate_skill_match_score(self, resume_skills: Dict[str, List[str]],
                                    jd_skills: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate detailed skill matching scores by category"""
        category_scores = {}

        for category, jd_category_skills in jd_skills.items():
            if not jd_category_skills:  # No skills required in this category
                category_scores[category] = 1.0
                continue

            resume_category_skills = resume_skills.get(category, [])

            # Calculate match percentage for this category
            matched_skills = set(resume_category_skills).intersection(set(jd_category_skills))
            match_percentage = len(matched_skills) / len(jd_category_skills)

            category_scores[category] = match_percentage

        return category_scores

    def calculate_experience_score(self, resume_experience: int, jd_text: str) -> float:
        """Calculate experience alignment score"""
        jd_lower = jd_text.lower()

        required_experience = 0

        # Look for experience patterns
        experience_patterns = [
            r'(\d+)\s*[-+]?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\s*[-+]?\s*years?',
            r'minimum\s*(\d+)\s*years?',
            r'(\d+)\s*to\s*(\d+)\s*years?',
            r'(\d+)\s*[-+]\s*years?',
        ]

        for pattern in experience_patterns:
            matches = re.findall(pattern, jd_lower)
            if matches:
                if isinstance(matches[0], tuple):  # Range like "2 to 5 years"
                    required_experience = int(matches[0][0])  # Take minimum
                else:
                    required_experience = int(matches[0])
                break

        # Check for experience level keywords
        experience_keywords = {
            'fresher': 0, 'entry level': 0, 'entry-level': 0, 'no experience': 0,
            'junior': 1, '1-2 years': 1, '0-2 years': 1,
            'mid level': 3, 'mid-level': 3, '2-5 years': 3,
            'senior': 5, '5+ years': 5, 'experienced': 5
        }

        if required_experience == 0:
            for level, years in experience_keywords.items():
                if level in jd_lower:
                    required_experience = years
                    break

        # Calculate score based on experience alignment
        if required_experience == 0:  # No specific requirement
            return 0.8  # Neutral score

        if resume_experience >= required_experience:
            # Meeting or exceeding requirements
            if resume_experience <= required_experience + 2:
                return 1.0  # Perfect match
            else:
                return 0.9  # Overqualified but still good
        else:
            # Under-qualified
            if resume_experience >= required_experience * 0.7:
                return 0.7  # Close enough
            else:
                return 0.4  # Significantly under-qualified

    def calculate_education_score(self, resume_education: str, jd_text: str) -> float:
        """Calculate education alignment score"""
        jd_lower = jd_text.lower()
        resume_edu_lower = resume_education.lower()

        # Get resume education level
        resume_level = 0
        for edu, level in self.education_levels.items():
            if edu in resume_edu_lower:
                resume_level = max(resume_level, level)

        # Extract required education from JD
        required_level = 0
        for edu, level in self.education_levels.items():
            if edu in jd_lower:
                required_level = max(required_level, level)

        # Calculate score
        if required_level == 0:  # No specific requirement
            return 0.8

        if resume_level >= required_level:
            return 1.0  # Meets or exceeds requirements
        elif resume_level >= required_level - 1:
            return 0.7  # Close match
        else:
            return 0.3  # Doesn't meet requirements

    def calculate_keyword_density_score(self, resume_text: str, jd_text: str) -> float:
        """Calculate keyword density and relevance score"""
        jd_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', jd_text.lower()))
        resume_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', resume_text.lower()))

        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
                      'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
                      'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'with', 'from', 'they',
                      'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just',
                      'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}

        jd_words = jd_words - stop_words
        resume_words = resume_words - stop_words

        if not jd_words:
            return 0.5

        # Calculate intersection
        common_words = jd_words.intersection(resume_words)
        keyword_score = len(common_words) / len(jd_words)

        return min(1.0, keyword_score * 1.2)  # Slight boost for keyword matching

    def analyze_domain_relevance(self, resume_text: str, jd_text: str) -> float:
        """Analyze domain-specific relevance"""
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()

        # Domain keywords
        domains = {
            'data_science': ['data science', 'data scientist', 'analytics', 'statistical', 'predictive modeling',
                             'data analysis'],
            'data_engineering': ['data engineer', 'etl', 'data pipeline', 'data warehouse', 'streaming', 'big data'],
            'machine_learning': ['machine learning', 'ml', 'ai', 'artificial intelligence', 'model', 'deep learning'],
            'software_development': ['software developer', 'programming', 'coding', 'development',
                                     'software engineering', 'full stack'],
            'business_intelligence': ['business intelligence', 'bi', 'reporting', 'dashboard', 'kpi', 'visualization']
        }

        # Find dominant domain in JD
        jd_domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in jd_lower)
            jd_domain_scores[domain] = score

        if not any(jd_domain_scores.values()):
            return 0.7  # No clear domain detected

        dominant_jd_domain = max(jd_domain_scores, key=jd_domain_scores.get)

        # Check resume alignment with dominant domain
        resume_domain_score = sum(1 for keyword in domains[dominant_jd_domain] if keyword in resume_lower)
        max_possible_score = len(domains[dominant_jd_domain])

        if max_possible_score == 0:
            return 0.7

        domain_alignment = resume_domain_score / max_possible_score
        return min(1.0, domain_alignment * 1.3)

    def generate_fallback_evaluation_hybrid(self, resume_text: str, jd_text: str,
                                            manual_experience: int = None, manual_education: str = None) -> Dict[
        str, any]:
        """Enhanced fallback evaluation with hybrid manual/extracted data"""

        try:
            self.logger.info("Generating hybrid fallback evaluation")

            # Extract detailed skills from both texts
            resume_skills = self.extract_detailed_skills(resume_text)
            jd_skills = self.extract_detailed_skills(jd_text)

            # Calculate various score components
            skill_scores = self.calculate_skill_match_score(resume_skills, jd_skills)
            keyword_density_score = self.calculate_keyword_density_score(resume_text, jd_text)
            domain_relevance_score = self.analyze_domain_relevance(resume_text, jd_text)

            # Extract data from resume
            extracted_experience = self.extract_experience_years(resume_text)
            extracted_education = self.extract_education_level(resume_text)

            # Resolve using hybrid approach
            final_experience, exp_source = self.resolve_experience_hybrid(extracted_experience, manual_experience)
            final_education, edu_source = self.resolve_education_hybrid(extracted_education, manual_education)

            # Calculate scores using resolved data
            experience_score = self.calculate_experience_score(final_experience, jd_text)
            education_score = self.calculate_education_score(final_education, jd_text)

            # Calculate weighted skill score
            weighted_skill_score = 0
            total_weight = 0

            for category, score in skill_scores.items():
                weight = self.skill_categories[category]['weight']
                weighted_skill_score += score * weight
                total_weight += weight

            if total_weight > 0:
                weighted_skill_score = weighted_skill_score / total_weight

            # Combine all scores with enhanced weighting
            base_score = (
                    weighted_skill_score * 0.35 +  # Skill matching (most important)
                    keyword_density_score * 0.25 +  # Keyword density
                    domain_relevance_score * 0.20 +  # Domain relevance
                    experience_score * 0.12 +  # Experience alignment
                    education_score * 0.08  # Education alignment
            )

            # Add small randomness for realistic variation
            import random
            variation = random.uniform(-0.03, 0.03)
            base_score += variation

            # Convert to percentage and apply bounds
            final_score = max(self.min_score, min(95, int(base_score * 100)))

            # Determine verdict
            if final_score >= 75:
                verdict = 'High'
            elif final_score >= 50:
                verdict = 'Medium'
            else:
                verdict = 'Low'

            # Identify missing skills
            missing_skills = []
            for category, jd_category_skills in jd_skills.items():
                resume_category_skills = resume_skills.get(category, [])
                missing = set(jd_category_skills) - set(resume_category_skills)
                missing_skills.extend(missing)

            # Generate intelligent feedback with hybrid data info
            feedback = self.generate_rule_based_feedback_hybrid(
                final_score, verdict, missing_skills, skill_scores,
                final_experience, exp_source, final_education, edu_source
            )

            # Generate recommendations
            recommendations = self.generate_recommendations(final_score, missing_skills)

            self.logger.info(f"Hybrid fallback evaluation completed: {final_score}/100 ({verdict})")
            self.logger.info(f"Used: {final_experience} years exp ({exp_source}), {final_education} edu ({edu_source})")

            return {
                'overall_score': final_score,
                'verdict': verdict,
                'component_scores': {
                    'skill_match': int(weighted_skill_score * 100),
                    'keyword_density': int(keyword_density_score * 100),
                    'domain_relevance': int(domain_relevance_score * 100),
                    'experience_match': int(experience_score * 100),
                    'education_match': int(education_score * 100)
                },
                'skill_category_scores': {cat: int(score * 100) for cat, score in skill_scores.items()},
                'missing_skills': missing_skills[:10],
                'feedback': feedback,
                'recommendations': recommendations,
                'strengths': self.identify_strengths(skill_scores, final_score),
                'weaknesses': self.identify_weaknesses(missing_skills, skill_scores),
                'llm_powered': False,
                'source': 'Enhanced Rule-based Analysis (Hybrid)',
                'resume_skills_found': resume_skills,
                'jd_skills_required': jd_skills,
                'hybrid_data': {
                    'experience_years': final_experience,
                    'experience_source': exp_source,
                    'education_level': final_education,
                    'education_source': edu_source,
                    'extracted_experience': extracted_experience,
                    'manual_experience': manual_experience,
                    'extracted_education': extracted_education,
                    'manual_education': manual_education
                }
            }

        except Exception as e:
            self.logger.error(f"Hybrid fallback evaluation failed: {e}")
            return {
                'overall_score': self.min_score,
                'verdict': 'Low',
                'feedback': f"Evaluation failed: {str(e)}",
                'llm_powered': False,
                'source': 'Error Fallback',
                'missing_skills': [],
                'recommendations': ["Please try again or contact support"]
            }

    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience from text"""
        text_lower = text.lower()

        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yr[s]?\s*(?:of\s*)?experience',
        ]

        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    years.append(int(match))
                except ValueError:
                    continue

        return max(years) if years else 0

    def extract_education_level(self, text: str) -> str:
        """Extract education level from text"""
        text_lower = text.lower()

        highest_level = 0
        found_education = 'not_specified'

        for edu, level in self.education_levels.items():
            if edu in text_lower and level > highest_level:
                highest_level = level
                found_education = edu

        return found_education

    def generate_rule_based_feedback_hybrid(self, score: int, verdict: str, missing_skills: List[str],
                                            skill_scores: Dict[str, float], experience_years: int, exp_source: str,
                                            education_level: str, edu_source: str) -> str:
        """Generate feedback with hybrid data transparency"""

        feedback_parts = []

        # Opening assessment
        if score >= 80:
            feedback_parts.append(
                "🎉 **Excellent Match!** Your resume demonstrates strong alignment with the job requirements.")
        elif score >= 65:
            feedback_parts.append("👍 **Good Match!** Your resume meets most of the key requirements for this position.")
        elif score >= 45:
            feedback_parts.append(
                "⚠️ **Moderate Match.** Your resume shows potential but needs improvement in key areas.")
        else:
            feedback_parts.append(
                "📈 **Needs Significant Improvement.** Your resume requires substantial development to match this role.")

        feedback_parts.append(f"\n**Overall Compatibility Score: {score}/100**")
        feedback_parts.append(f"**Match Level: {verdict}**\n")

        # Hybrid data transparency
        feedback_parts.append("### 📊 **Profile Analysis:**")
        feedback_parts.append(f"**Experience:** {experience_years} years ({exp_source})")
        feedback_parts.append(f"**Education:** {education_level.replace('_', ' ').title()} ({edu_source})")
        feedback_parts.append("")

        # Skill category analysis
        feedback_parts.append("### 🔍 **Skill Category Analysis:**")

        strong_categories = []
        weak_categories = []

        for category, score_pct in skill_scores.items():
            category_name = category.replace('_', ' ').title()
            score_val = int(score_pct * 100)

            if score_pct >= 0.7:
                strong_categories.append(f"✅ **{category_name}**: {score_val}% match")
            elif score_pct >= 0.3:
                weak_categories.append(f"⚠️ **{category_name}**: {score_val}% match - needs strengthening")
            else:
                weak_categories.append(f"❌ **{category_name}**: {score_val}% match - requires development")

        if strong_categories:
            feedback_parts.append("**Strengths:**")
            feedback_parts.extend(strong_categories)
            feedback_parts.append("")

        if weak_categories:
            feedback_parts.append("**Areas for Improvement:**")
            feedback_parts.extend(weak_categories)
            feedback_parts.append("")

        # Priority skills to develop
        if missing_skills:
            feedback_parts.append("### 🎯 **Priority Skills to Develop:**")
            for skill in missing_skills[:8]:
                feedback_parts.append(f"• **{skill.title()}**")
            feedback_parts.append("")

        # Action items based on score
        feedback_parts.append("### 🚀 **Recommended Actions:**")

        if score >= 70:
            feedback_parts.extend([
                "1. **Apply with confidence** - You meet most requirements",
                "2. **Highlight relevant projects** in your cover letter",
                "3. **Prepare examples** of your key accomplishments",
                "4. **Emphasize your strongest skill areas** during interviews"
            ])
        elif score >= 50:
            feedback_parts.extend([
                "1. **Address skill gaps** through online courses or projects",
                "2. **Strengthen your resume** with relevant keywords",
                "3. **Consider gaining experience** in missing areas",
                "4. **Build a portfolio** showcasing relevant work"
            ])
        else:
            feedback_parts.extend([
                "1. **Focus on developing core skills** mentioned in the job",
                "2. **Build a portfolio** demonstrating relevant capabilities",
                "3. **Consider additional training** or certification programs",
                "4. **Gain practical experience** through internships or projects"
            ])

        return "\n".join(feedback_parts)

    def generate_recommendations(self, score: int, missing_skills: List[str]) -> List[str]:
        """Generate actionable recommendations based on score and missing skills"""
        recommendations = []

        if score >= 75:
            recommendations.extend([
                "Strong candidate - proceed with interview process",
                "Highlight your relevant experience and projects",
                "Prepare for technical discussions in your strong areas"
            ])
        elif score >= 55:
            recommendations.extend([
                "Good potential - consider with some reservations",
                f"Focus on developing: {', '.join(missing_skills[:3])}" if missing_skills else "Continue building on existing strengths",
                "May benefit from additional screening or assessment"
            ])
        else:
            recommendations.extend([
                "Significant skill development needed before applying",
                f"Critical missing skills: {', '.join(missing_skills[:5])}" if missing_skills else "Overall skill alignment needs improvement",
                "Consider gaining relevant experience through projects or courses"
            ])

        return recommendations

    def identify_strengths(self, skill_scores: Dict[str, float], overall_score: int) -> List[str]:
        """Identify candidate strengths based on analysis"""
        strengths = []

        for category, score in skill_scores.items():
            if score >= 0.7:
                category_name = category.replace('_', ' ').title()
                strengths.append(f"Strong {category_name} skills")

        if overall_score >= 70:
            strengths.append("Good overall alignment with job requirements")

        if not strengths:
            strengths.append("Basic qualifications present")

        return strengths[:5]  # Limit to top 5 strengths

    def identify_weaknesses(self, missing_skills: List[str], skill_scores: Dict[str, float]) -> List[str]:
        """Identify areas for improvement"""
        weaknesses = []

        # Add skill category weaknesses
        for category, score in skill_scores.items():
            if score < 0.3:
                category_name = category.replace('_', ' ').title()
                weaknesses.append(f"Limited {category_name} experience")

        # Add specific missing skills
        if missing_skills:
            weaknesses.append(f"Missing key skills: {', '.join(missing_skills[:3])}")

        if not weaknesses:
            weaknesses.append("Minor gaps in specific requirements")

        return weaknesses[:4]  # Limit to top 4 weaknesses

    # Legacy methods for compatibility
    def calculate_comprehensive_score(self, **kwargs) -> Dict[str, any]:
        """Legacy method for backward compatibility"""
        resume_text = kwargs.get('resume_text', '')
        jd_text = kwargs.get('jd_text', '')
        manual_experience = kwargs.get('manual_experience')
        manual_education = kwargs.get('manual_education')

        return self.generate_fallback_evaluation_hybrid(resume_text, jd_text, manual_experience, manual_education)


# Global instances
llm_integrated_scorer = LLMIntegratedScorer()

# Legacy compatibility
relevance_scorer = llm_integrated_scorer