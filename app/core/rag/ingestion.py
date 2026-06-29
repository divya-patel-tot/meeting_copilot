import fitz
from docx import Document


def parse_pdf(file_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def parse_docx(file_path: str) -> str:
    """Extract all paragraph text from a Word document."""
    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)
