"""
Keyword matching service using TF-IDF, BM25, and fuzzy matching (approved algorithms)
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fuzzywuzzy import fuzz, process
from typing import Dict, List, Tuple, Set
import re
import logging


class KeywordMatcher:
    """Advanced keyword matching using multiple algorithms"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def exact_keyword_match(self, resume_text: str, jd_text: str) -> Dict[str, any]:
        """Exact keyword matching between resume and job description"""
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()

        # Extract keywords from JD (simple word extraction)
        jd_keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', jd_lower))

        # Find matches in resume
        matched_keywords = set()
        for keyword in jd_keywords:
            if keyword in resume_lower:
                matched_keywords.add(keyword)

        # Calculate match percentage
        match_percentage = len(matched_keywords) / len(jd_keywords) * 100 if jd_keywords else 0

        return {
            'matched_keywords': matched_keywords,
            'total_jd_keywords': len(jd_keywords),
            'matched_count': len(matched_keywords),
            'match_percentage': round(match_percentage, 2),
            'missing_keywords': jd_keywords - matched_keywords
        }

    def tfidf_similarity(self, resume_text: str, jd_text: str) -> Dict[str, any]:
        """TF-IDF based similarity scoring"""
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),  # Include unigrams and bigrams
                max_features=1000,
                min_df=1,
                lowercase=True
            )

            # Fit and transform both texts
            documents = [resume_text, jd_text]
            tfidf_matrix = vectorizer.fit_transform(documents)

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_score = similarity_matrix[0][1]  # Resume vs JD similarity

            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            resume_scores = tfidf_matrix[0].toarray()[0]
            jd_scores = tfidf_matrix[1].toarray()[0]

            # Find top matching terms
            common_terms = []
            for i, feature in enumerate(feature_names):
                if resume_scores[i] > 0 and jd_scores[i] > 0:
                    common_terms.append({
                        'term': feature,
                        'resume_score': round(resume_scores[i], 4),
                        'jd_score': round(jd_scores[i], 4),
                        'combined_score': round(resume_scores[i] * jd_scores[i], 4)
                    })

            # Sort by combined score
            common_terms.sort(key=lambda x: x['combined_score'], reverse=True)

            return {
                'similarity_score': round(similarity_score, 4),
                'similarity_percentage': round(similarity_score * 100, 2),
                'common_terms': common_terms[:20],  # Top 20 common terms
                'total_features': len(feature_names)
            }

        except Exception as e:
            self.logger.error(f"TF-IDF similarity calculation failed: {str(e)}")
            return {
                'similarity_score': 0.0,
                'similarity_percentage': 0.0,
                'common_terms': [],
                'total_features': 0,
                'error': str(e)
            }

    def bm25_similarity(self, resume_text: str, jd_text: str) -> Dict[str, any]:
        """BM25 similarity (using TF-IDF with BM25-like parameters)"""
        try:
            # BM25-inspired TF-IDF with different parameters
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000,
                min_df=1,
                sublinear_tf=True,  # BM25-like TF normalization
                lowercase=True,
                norm='l2'
            )

            documents = [resume_text, jd_text]
            tfidf_matrix = vectorizer.fit_transform(documents)

            # Calculate similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            bm25_score = similarity_matrix[0][1]

            return {
                'bm25_score': round(bm25_score, 4),
                'bm25_percentage': round(bm25_score * 100, 2),
                'algorithm': 'BM25-inspired TF-IDF'
            }

        except Exception as e:
            self.logger.error(f"BM25 similarity calculation failed: {str(e)}")
            return {
                'bm25_score': 0.0,
                'bm25_percentage': 0.0,
                'error': str(e)
            }

    def fuzzy_skill_matching(self, resume_skills: Set[str], jd_skills: Set[str],
                             threshold: int = 80) -> Dict[str, any]:
        """Fuzzy matching for skills using fuzzywuzzy"""
        if not resume_skills or not jd_skills:
            return {
                'exact_matches': set(),
                'fuzzy_matches': [],
                'missing_skills': jd_skills,
                'match_percentage': 0.0
            }

        resume_skills_list = list(resume_skills)
        jd_skills_list = list(jd_skills)

        exact_matches = resume_skills.intersection(jd_skills)
        fuzzy_matches = []
        matched_jd_skills = set(exact_matches)

        # Find fuzzy matches for remaining JD skills
        remaining_jd_skills = jd_skills - exact_matches

        for jd_skill in remaining_jd_skills:
            # Find best match in resume skills
            match = process.extractOne(
                jd_skill,
                resume_skills_list,
                scorer=fuzz.token_set_ratio
            )

            if match and match[1] >= threshold:
                fuzzy_matches.append({
                    'jd_skill': jd_skill,
                    'resume_skill': match[0],
                    'similarity': match[1]
                })
                matched_jd_skills.add(jd_skill)

        # Calculate match percentage
        total_matches = len(exact_matches) + len(fuzzy_matches)
        match_percentage = (total_matches / len(jd_skills)) * 100 if jd_skills else 0

        return {
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches,
            'missing_skills': jd_skills - matched_jd_skills,
            'match_percentage': round(match_percentage, 2),
            'total_jd_skills': len(jd_skills),
            'total_matches': total_matches
        }

    def comprehensive_keyword_matching(self, resume_text: str, jd_text: str,
                                       resume_skills: Set[str] = None,
                                       jd_skills: Set[str] = None) -> Dict[str, any]:
        """Comprehensive matching using all algorithms"""

        # Exact keyword matching
        exact_match = self.exact_keyword_match(resume_text, jd_text)

        # TF-IDF similarity
        tfidf_result = self.tfidf_similarity(resume_text, jd_text)

        # BM25 similarity
        bm25_result = self.bm25_similarity(resume_text, jd_text)

        # Fuzzy skill matching (if skills provided)
        fuzzy_result = {}
        if resume_skills and jd_skills:
            fuzzy_result = self.fuzzy_skill_matching(resume_skills, jd_skills)

        # Calculate combined keyword score
        keyword_scores = [
            exact_match['match_percentage'] / 100,
            tfidf_result['similarity_score'],
            bm25_result['bm25_score']
        ]

        # Add fuzzy skill score if available
        if fuzzy_result:
            keyword_scores.append(fuzzy_result['match_percentage'] / 100)

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2] if fuzzy_result else [0.4, 0.4, 0.2]
        combined_score = sum(score * weight for score, weight in zip(keyword_scores, weights))

        return {
            'combined_keyword_score': round(combined_score, 4),
            'combined_percentage': round(combined_score * 100, 2),
            'exact_matching': exact_match,
            'tfidf_matching': tfidf_result,
            'bm25_matching': bm25_result,
            'fuzzy_skill_matching': fuzzy_result,
            'algorithm_weights': dict(zip(['exact', 'tfidf', 'bm25', 'fuzzy'], weights))
        }


# Global keyword matcher instance
keyword_matcher = KeywordMatcher()