"""
Document Processor Service.
Handles PDF reading, text extraction, cleaning, and section segmentation.
"""

import io
import re
from typing import Dict, Any

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

class DocumentProcessorService:
    @staticmethod
    def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
        """Extracts text from PDF binary content."""
        if not PYPDF_AVAILABLE:
            # Fallback mock/plain text extractor if pypdf is not installed
            return pdf_bytes.decode('utf-8', errors='ignore')

        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            text_pages = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_pages.append(extracted)
            return "\n".join(text_pages)
        except Exception as e:
            print(f"[DocumentProcessor] PDF Parsing error: {e}")
            return pdf_bytes.decode('utf-8', errors='ignore')

    @staticmethod
    def clean_medical_text(raw_text: str) -> str:
        """Normalizes whitespace and removes unwanted artifacts."""
        if not raw_text:
            return ""
        # Remove multiple newlines/spaces
        cleaned = re.sub(r'\r\n|\r', '\n', raw_text)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        return cleaned.strip()

    @staticmethod
    def segment_sections(text: str) -> Dict[str, str]:
        """Segments report into standard clinical sections if present."""
        sections = {
            "history": "",
            "findings": "",
            "impression": "",
            "full_text": text
        }
        
        # Simple section header matching
        history_match = re.search(r'(?:history|clinical history|presentation):\s*(.*?)(?=\n[A-Z]|\Z)', text, re.IGNORECASE | re.DOTALL)
        if history_match:
            sections["history"] = history_match.group(1).strip()
            
        findings_match = re.search(r'(?:findings|examination|results):\s*(.*?)(?=\n[A-Z]|\Z)', text, re.IGNORECASE | re.DOTALL)
        if findings_match:
            sections["findings"] = findings_match.group(1).strip()
            
        impression_match = re.search(r'(?:impression|assessment|plan|conclusion):\s*(.*?)(?=\n[A-Z]|\Z)', text, re.IGNORECASE | re.DOTALL)
        if impression_match:
            sections["impression"] = impression_match.group(1).strip()

        return sections
