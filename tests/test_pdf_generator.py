"""
Unit tests for the ATS-Optimized PDF Generator.
Verifies PDF binary generation, unicode sanitization, and rewritten bullet persistence.
"""

import io
import unittest
from pypdf import PdfReader
from src.pdf_generator import build_ats_cv_pdf, sanitize_latin1


class TestPDFGenerator(unittest.TestCase):

    def test_sanitize_latin1(self):
        text = "“Smart quotes”, ‘single quotes’, em—dash, en–dash, and • bullet."
        sanitized = sanitize_latin1(text)
        self.assertNotIn("“", sanitized)
        self.assertNotIn("•", sanitized)
        self.assertIn("Smart quotes", sanitized)
        # Verify it encodes cleanly to latin-1
        sanitized.encode("latin-1")

    def test_build_ats_cv_pdf_basic(self):
        payload = {
            "name": "Aqsa Ehtesham",
            "role": "Frontend Web Developer",
            "location": "Lahore, Pakistan",
            "email": "aqsa@example.com",
            "phone": "+92 310 4577412",
            "summary": "Experienced engineer specializing in high-performance web applications and design systems.",
            "skillsDetected": ["React", "JavaScript", "TypeScript", "TailwindCSS"],
            "bullets": [
                {
                    "text": "Spearheaded frontend architecture modernization across 4 high-traffic web applications, improving load times by 38% for 450K+ monthly active users.",
                    "is_weak": False
                },
                {
                    "text": "Engineered automated regression testing pipeline, reducing production defect escape rate by 24%.",
                    "is_weak": False
                }
            ]
        }
        pdf_bytes = build_ats_cv_pdf(payload)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 1000)

        # Parse with pypdf and verify content text
        reader = PdfReader(io.BytesIO(pdf_bytes))
        self.assertGreaterEqual(len(reader.pages), 1)
        full_text = "".join(page.extract_text() for page in reader.pages)
        self.assertIn("AQSA EHTESHAM", full_text)
        self.assertIn("Frontend Web Developer", full_text)
        self.assertIn("Spearheaded frontend architecture modernization", full_text)
        self.assertIn("38%", full_text)
        self.assertIn("regression testing pipeline", full_text)

    def test_build_ats_cv_pdf_empty_fallback(self):
        # Empty payload should still generate valid PDF without error
        pdf_bytes = build_ats_cv_pdf({})
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
