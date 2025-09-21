"""
PDF parsing service using PyMuPDF and pdfplumber (approved packages)
"""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import Dict, Optional
import logging


class PDFParser:
    """Parse PDF files to extract text content"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text_pymupdf(self, file_path: str) -> Dict[str, any]:
        """Extract text using PyMuPDF (primary method)"""
        try:
            doc = fitz.open(file_path)
            text_content = ""
            metadata = {
                'total_pages': len(doc),
                'method_used': 'PyMuPDF',
                'success': False,
                'error': None
            }

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()
                text_content += "\n"

            doc.close()

            if text_content.strip():
                metadata['success'] = True
                return {
                    'text': text_content.strip(),
                    'metadata': metadata
                }
            else:
                metadata['error'] = "No text extracted"
                return {'text': "", 'metadata': metadata}

        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {str(e)}")
            return {
                'text': "",
                'metadata': {
                    'total_pages': 0,
                    'method_used': 'PyMuPDF',
                    'success': False,
                    'error': str(e)
                }
            }

    def extract_text_pdfplumber(self, file_path: str) -> Dict[str, any]:
        """Extract text using pdfplumber (fallback method)"""
        try:
            text_content = ""
            metadata = {
                'total_pages': 0,
                'method_used': 'pdfplumber',
                'success': False,
                'error': None
            }

            with pdfplumber.open(file_path) as pdf:
                metadata['total_pages'] = len(pdf.pages)

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text
                        text_content += "\n"

            if text_content.strip():
                metadata['success'] = True
                return {
                    'text': text_content.strip(),
                    'metadata': metadata
                }
            else:
                metadata['error'] = "No text extracted"
                return {'text': "", 'metadata': metadata}

        except Exception as e:
            self.logger.error(f"pdfplumber extraction failed: {str(e)}")
            return {
                'text': "",
                'metadata': {
                    'total_pages': 0,
                    'method_used': 'pdfplumber',
                    'success': False,
                    'error': str(e)
                }
            }

    def extract_text(self, file_path: str) -> Dict[str, any]:
        """Extract text using primary method, fallback to secondary if needed"""
        if not Path(file_path).exists():
            return {
                'text': "",
                'metadata': {
                    'success': False,
                    'error': "File does not exist"
                }
            }

        # Try PyMuPDF first (primary)
        result = self.extract_text_pymupdf(file_path)

        if result['metadata']['success'] and result['text'].strip():
            return result

        # Fallback to pdfplumber
        self.logger.info("PyMuPDF failed, trying pdfplumber fallback")
        fallback_result = self.extract_text_pdfplumber(file_path)

        # Combine metadata from both attempts
        fallback_result['metadata']['primary_method_error'] = result['metadata'].get('error')

        return fallback_result


# Global PDF parser instance
pdf_parser = PDFParser()