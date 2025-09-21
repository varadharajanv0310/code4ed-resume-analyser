"""
Unified file parser that handles both PDF and DOCX files
"""
from pathlib import Path
from typing import Dict, Optional
import logging

from .pdf_parser import pdf_parser
from .docx_parser import docx_parser


class FileParser:
    """Main file parsing service that routes to appropriate parser"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_extensions = {'.pdf', '.docx'}

    def is_supported_file(self, file_path: str) -> bool:
        """Check if file extension is supported"""
        return Path(file_path).suffix.lower() in self.supported_extensions

    def get_file_type(self, file_path: str) -> str:
        """Get file type from extension"""
        return Path(file_path).suffix.lower().replace('.', '')

    def parse_file(self, file_path: str) -> Dict[str, any]:
        """Parse file and extract text content"""
        if not self.is_supported_file(file_path):
            return {
                'text': "",
                'metadata': {
                    'success': False,
                    'error': f"Unsupported file type. Supported: {self.supported_extensions}"
                }
            }

        file_type = self.get_file_type(file_path)

        try:
            if file_type == 'pdf':
                result = pdf_parser.extract_text(file_path)
            elif file_type == 'docx':
                result = docx_parser.extract_text(file_path)
            else:
                return {
                    'text': "",
                    'metadata': {
                        'success': False,
                        'error': f"Unsupported file type: {file_type}"
                    }
                }

            # Add file info to metadata
            file_path_obj = Path(file_path)
            result['metadata']['file_name'] = file_path_obj.name
            result['metadata']['file_size'] = file_path_obj.stat().st_size
            result['metadata']['file_type'] = file_type

            return result

        except Exception as e:
            self.logger.error(f"File parsing failed for {file_path}: {str(e)}")
            return {
                'text': "",
                'metadata': {
                    'success': False,
                    'error': f"File parsing failed: {str(e)}",
                    'file_name': Path(file_path).name,
                    'file_type': file_type
                }
            }


# Global file parser instance
file_parser = FileParser()