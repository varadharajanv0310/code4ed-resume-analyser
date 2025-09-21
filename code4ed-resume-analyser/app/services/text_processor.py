"""
Text processing service using spaCy and NLTK (approved packages)
"""
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from typing import List, Dict, Set, Tuple
import logging


class TextProcessor:
    """Advanced text processing using spaCy and NLTK"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            self.logger.error("NLTK data not found. Run the NLTK download commands.")
            self.stop_words = set()
            self.lemmatizer = None

        # Common skills keywords for extraction
        self.tech_skills = {
            'python', 'java', 'javascript', 'react', 'angular', 'nodejs', 'django', 'flask',
            'fastapi', 'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'git', 'jenkins', 'ci/cd', 'machine learning', 'deep learning',
            'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'html', 'css', 'bootstrap',
            'rest api', 'microservices', 'agile', 'scrum', 'linux', 'windows', 'hadoop', 'spark'
        }

    def clean_text(self, text: str) -> str:
        """Basic text cleaning and normalization"""
        if not text:
            return ""

        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep letters, numbers, and basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)

        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract common resume sections using pattern matching"""
        sections = {
            'personal_info': '',
            'objective': '',
            'education': '',
            'experience': '',
            'skills': '',
            'projects': '',
            'certifications': ''
        }

        text_lower = text.lower()

        # Define section patterns
        patterns = {
            'objective': r'(objective|summary|profile|about)(.*?)(?=education|experience|skills|projects|$)',
            'education': r'(education|academic|qualification)(.*?)(?=experience|skills|projects|objective|$)',
            'experience': r'(experience|employment|work history|professional)(.*?)(?=education|skills|projects|objective|$)',
            'skills': r'(skills|technologies|technical skills|competencies)(.*?)(?=education|experience|projects|objective|$)',
            'projects': r'(projects|portfolio|work samples)(.*?)(?=education|experience|skills|objective|$)',
            'certifications': r'(certifications|certificates|achievements|awards)(.*?)(?=education|experience|skills|projects|$)'
        }

        for section, pattern in patterns.items():
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            if match:
                sections[section] = match.group(2).strip()

        return sections

    def extract_skills_spacy(self, text: str) -> Set[str]:
        """Extract skills using spaCy NER and pattern matching"""
        skills = set()

        if not self.nlp:
            return self.extract_skills_nltk(text)

        doc = self.nlp(text.lower())

        # Extract using named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:  # Organizations and products often represent technologies
                skill_text = ent.text.lower().strip()
                if len(skill_text) > 2 and skill_text in self.tech_skills:
                    skills.add(skill_text)

        # Extract using noun chunks and token matching
        for token in doc:
            if token.text.lower() in self.tech_skills:
                skills.add(token.text.lower())

        # Pattern-based extraction for multi-word skills
        for skill in self.tech_skills:
            if skill in text.lower():
                skills.add(skill)

        return skills

    def extract_skills_nltk(self, text: str) -> Set[str]:
        """Extract skills using NLTK as fallback"""
        skills = set()
        text_lower = text.lower()

        # Tokenize and check against skill list
        if self.lemmatizer:
            words = word_tokenize(text_lower)
            for word in words:
                if word in self.tech_skills:
                    skills.add(word)

                # Check lemmatized form
                lemmatized = self.lemmatizer.lemmatize(word)
                if lemmatized in self.tech_skills:
                    skills.add(lemmatized)

        # Pattern-based extraction for multi-word skills
        for skill in self.tech_skills:
            if skill in text_lower:
                skills.add(skill)

        return skills

    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience using pattern matching"""
        text_lower = text.lower()

        # Patterns for experience extraction
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:work\s*)?(?:professional\s*)?experience',
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
        """Extract highest education level"""
        text_lower = text.lower()

        education_levels = {
            'phd': ['phd', 'ph.d', 'doctorate', 'doctoral'],
            'masters': ['masters', 'master', 'm.s', 'msc', 'm.sc', 'mba', 'm.b.a', 'me', 'm.e'],
            'bachelors': ['bachelors', 'bachelor', 'b.s', 'bsc', 'b.sc', 'be', 'b.e', 'b.tech', 'btech'],
            'diploma': ['diploma', 'certificate'],
            'high_school': ['high school', '12th', 'intermediate', 'higher secondary']
        }

        for level, keywords in education_levels.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return level

        return 'not_specified'

    def tokenize_and_normalize(self, text: str) -> List[str]:
        """Tokenize and normalize text for matching"""
        if not text:
            return []

        # Clean text first
        cleaned = self.clean_text(text)

        if self.nlp:
            # Use spaCy for better tokenization
            doc = self.nlp(cleaned.lower())
            tokens = []

            for token in doc:
                if (not token.is_stop and
                        not token.is_punct and
                        not token.is_space and
                        len(token.text) > 2):
                    tokens.append(token.lemma_)

            return tokens

        else:
            # Fallback to NLTK
            tokens = word_tokenize(cleaned.lower())

            if self.lemmatizer and self.stop_words:
                normalized = []
                for token in tokens:
                    if (token not in self.stop_words and
                            token not in string.punctuation and
                            len(token) > 2):
                        normalized.append(self.lemmatizer.lemmatize(token))
                return normalized
            else:
                return [t for t in tokens if len(t) > 2 and t not in string.punctuation]

    def extract_keywords_tfidf(self, text: str, max_features: int = 50) -> List[Tuple[str, float]]:
        """Extract important keywords using TF-IDF"""
        try:
            # Clean and tokenize
            cleaned_text = self.clean_text(text)

            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=1
            )

            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Create keyword-score pairs
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)

            return keyword_scores

        except Exception as e:
            self.logger.error(f"TF-IDF extraction failed: {str(e)}")
            return []

    def process_text(self, text: str) -> Dict[str, any]:
        """Complete text processing pipeline"""
        if not text:
            return {
                'cleaned_text': '',
                'sections': {},
                'skills': set(),
                'experience_years': 0,
                'education_level': 'not_specified',
                'tokens': [],
                'keywords': []
            }

        # Process the text
        cleaned_text = self.clean_text(text)
        sections = self.extract_sections(text)
        skills = self.extract_skills_spacy(text)
        experience_years = self.extract_experience_years(text)
        education_level = self.extract_education_level(text)
        tokens = self.tokenize_and_normalize(text)
        keywords = self.extract_keywords_tfidf(text)

        return {
            'cleaned_text': cleaned_text,
            'sections': sections,
            'skills': skills,
            'experience_years': experience_years,
            'education_level': education_level,
            'tokens': tokens,
            'keywords': keywords
        }


# Global text processor instance
text_processor = TextProcessor()