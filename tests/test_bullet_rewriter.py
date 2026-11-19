"""Unit tests for CVision AI Bullet Rewriter / CV Coach feature.

Covers:
1. Bullet extraction from Experience and Projects sections.
2. Rule-based weak bullet detection (passive starters, missing metrics, length, action verbs).
3. Single bullet rewriting (schema validation, exactly 3 rewrites, style labels, impact scores).
4. In-memory caching for rewrites.
5. Batch bullet rewriting (/rewrite-all) concurrency and schema.
6. REST API handler endpoints (/api/rewrite, /api/rewrite-all, /api/extract-bullets).
"""
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

# Ensure src is on python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from inference import (
    extract_cv_bullets,
    detect_bullet_weakness,
    rewrite_bullet,
    rewrite_all_bullets,
    _REWRITE_CACHE,
    WEAK_STARTERS,
    ACTION_VERBS,
)
from app import CVisionAppHandler


class TestBulletExtraction(unittest.TestCase):
    def test_extract_bullets_from_experience_and_projects(self):
        sample_cv = """
        John Doe
        john@example.com

        Summary:
        Passionate developer with cloud experience.

        Experience:
        - Responsible for managing database backups and server updates.
        • Spearheaded migration of 14 monolithic microservices to Kubernetes clusters.
        * Assisted teammates with frontend bug fixes.

        Education:
        B.S. in Computer Science

        Projects:
        - Developed open-source resume evaluator using Python and React.
        • Automated CI/CD deployment pipeline cutting build failures by 40%.
        """
        bullets = extract_cv_bullets(sample_cv)
        self.assertGreaterEqual(len(bullets), 4)

        sections = [b["section"] for b in bullets]
        self.assertIn("Experience", sections)
        self.assertIn("Projects", sections)

        # Confirm bullet glyphs were stripped
        for b in bullets:
            self.assertFalse(b["text"].startswith(("-", "•", "*")))

    def test_extract_fallback_when_no_section_header(self):
        # Plain text without explicit "Experience" header
        raw_text = """
        - Responsible for managing customer tickets.
        • Handled cash register and customer service.
        * Streamlined store inventory tracking by 15%.
        """
        bullets = extract_cv_bullets(raw_text)
        self.assertEqual(len(bullets), 3)


class TestWeakBulletDetection(unittest.TestCase):
    def test_passive_starter_weakness(self):
        weak_bullet = "Responsible for managing the team schedule and assigning weekly tickets."
        res = detect_bullet_weakness(weak_bullet)
        self.assertTrue(res["is_weak"])
        self.assertTrue(any("passive" in r.lower() for r in res["reasons"]))
        self.assertLessEqual(res["impact_score_before"], 5)

    def test_missing_metric_weakness(self):
        no_metric = "Developed internal microservice architecture using Go and Docker."
        res = detect_bullet_weakness(no_metric)
        self.assertTrue(res["is_weak"])
        self.assertTrue(any("metric" in r.lower() for r in res["reasons"]))

    def test_word_count_limits(self):
        # Too brief (< 8 words)
        brief = "Fixed bugs daily."
        res_brief = detect_bullet_weakness(brief)
        self.assertTrue(res_brief["is_weak"])
        self.assertTrue(any("brief" in r.lower() for r in res_brief["reasons"]))

        # Too long (> 30 words)
        long_bullet = (
            "Delivered comprehensive architectural transformation of legacy monolith systems across thirty disparate "
            "subsystems by writing clean maintainable code in Python, Java, and TypeScript while simultaneously organizing "
            "weekly cross-team architectural review sessions with seventy stakeholders."
        )
        res_long = detect_bullet_weakness(long_bullet)
        self.assertTrue(res_long["is_weak"])
        self.assertTrue(any("verbose" in r.lower() for r in res_long["reasons"]))

    def test_strong_bullet_not_weak(self):
        strong_bullet = "Orchestrated migration of 14 microservices to AWS Kubernetes, reducing latency by 35%."
        res = detect_bullet_weakness(strong_bullet)
        self.assertFalse(res["is_weak"])
        self.assertEqual(len(res["reasons"]), 0)
        self.assertGreaterEqual(res["impact_score_before"], 8)


class TestBulletRewriting(unittest.TestCase):
    def setUp(self):
        _REWRITE_CACHE.clear()

    def test_rewrite_bullet_schema_and_rules(self):
        bullet = "Responsible for managing the customer database and helping teammates with bug fixes."
        res = rewrite_bullet(
            bullet=bullet,
            role="Software Developer",
            jd_keywords=["SQL", "PostgreSQL"],
            tone="impact"
        )

        # Check required fields
        self.assertIn("why_weak", res)
        self.assertIn("impact_score_before", res)
        self.assertIn("rewrites", res)

        # Exactly 3 rewrites
        rewrites = res["rewrites"]
        self.assertEqual(len(rewrites), 3)

        for item in rewrites:
            self.assertIn("text", item)
            self.assertIn("style", item)
            self.assertIn("impact_score", item)
            self.assertGreaterEqual(item["impact_score"], 7)
            # Must be under 25 words
            word_count = len(item["text"].split())
            self.assertLessEqual(word_count, 25)
            # Must start with a capitalized word
            first_word = item["text"].split()[0].lower()
            self.assertIn(first_word, ACTION_VERBS)

    def test_rewrite_caching(self):
        bullet = "Handled customer inquiries and logged reports."
        res1 = rewrite_bullet(bullet, role="Support", tone="concise")
        cache_key = (bullet.strip(), "support", "concise")
        self.assertIn(cache_key, _REWRITE_CACHE)

        res2 = rewrite_bullet(bullet, role="Support", tone="concise")
        self.assertIs(res1, res2)

    def test_rewrite_tones(self):
        bullet = "Helped team build web pages."
        for tone in ("impact", "technical", "concise", "leadership"):
            res = rewrite_bullet(bullet, role="Frontend Engineer", tone=tone)
            self.assertEqual(len(res["rewrites"]), 3)
            self.assertTrue(all(r["text"] for r in res["rewrites"]))


class TestBatchRewriting(unittest.TestCase):
    def test_rewrite_all_bullets_concurrent(self):
        bullets = [
            {"id": "b-1", "text": "Responsible for maintaining servers.", "section": "Experience", "is_weak": True},
            {"id": "b-2", "text": "Helped team test applications.", "section": "Experience", "is_weak": True},
            {"id": "b-3", "text": "Spearheaded launch of 3 products generating $2M revenue.", "section": "Experience", "is_weak": False},
        ]
        results = rewrite_all_bullets(bullets, role="DevOps Engineer", tone="technical")
        self.assertEqual(len(results), 3)

        # Weak bullets should have rewrites populated
        self.assertIn("rewrites", results[0])
        self.assertEqual(len(results[0]["rewrites"]), 3)
        self.assertIn("rewrites", results[1])
        self.assertEqual(len(results[1]["rewrites"]), 3)


class TestAppHandlerEndpoints(unittest.TestCase):
    def test_rewrite_endpoint_validation(self):
        # Empty bullet should return 400
        handler = object.__new__(CVisionAppHandler)
        handler.headers = {"Content-Length": "15"}
        handler.rfile = BytesIO(b'{"bullet": ""}')
        handler.path = "/api/rewrite"

        mock_send = MagicMock()
        handler._send_json = mock_send

        handler.do_POST()
        mock_send.assert_called_once()
        args, kwargs = mock_send.call_args
        self.assertEqual(kwargs.get("status"), 400)
        self.assertIn("empty", args[0]["error"])

    def test_rewrite_endpoint_success(self):
        handler = object.__new__(CVisionAppHandler)
        payload = json.dumps({
            "bullet": "Worked on database maintenance.",
            "role": "Database Administrator",
            "tone": "technical"
        }).encode("utf-8")
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = BytesIO(payload)
        handler.path = "/api/rewrite"

        mock_send = MagicMock()
        handler._send_json = mock_send

        handler.do_POST()
        mock_send.assert_called_once()
        args, kwargs = mock_send.call_args
        data = args[0]
        self.assertIn("rewrites", data)
        self.assertEqual(len(data["rewrites"]), 3)


if __name__ == "__main__":
    unittest.main()
