import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> tuple[bool, str]:
    pdf_sentence = "The meeting responder PDF test sentence works."
    docx_sentence = "The meeting responder DOCX test sentence works."

    try:
        import fitz
        from docx import Document

        from app.core.rag.ingestion import parse_docx, parse_pdf

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), pdf_sentence)
            doc.save(str(pdf_path))
            doc.close()

            pdf_text = parse_pdf(str(pdf_path))
            if pdf_sentence not in pdf_text:
                return False, f"PDF sentence not found in: {pdf_text!r}"

            docx_path = Path(tmpdir) / "test.docx"
            word_doc = Document()
            word_doc.add_paragraph(docx_sentence)
            word_doc.save(str(docx_path))

            docx_text = parse_docx(str(docx_path))
            if docx_sentence not in docx_text:
                return False, f"DOCX sentence not found in: {docx_text!r}"

        return True, "PDF and DOCX parsing both returned expected sentences"
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    success, message = run()
    status = "PASS" if success else "FAIL"
    print(f"{status}: {message}")
