"""Unified full-stack web application server for CVision.

Provides:
- Serving of the React + Tailwind frontend dashboard (static/index.html).
- REST API endpoint 'POST /api/analyze' for live PDF / text resume inference.
- Healthcheck endpoint 'GET /api/health'.
- Automatic port fallback (8000 -> 8080) and automatic browser launch for live viva demo.
"""
import email.parser
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
import sys
import urllib.parse
import webbrowser

# Add src to Python path for seamless inference engine imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import (  # noqa: E402
    CVisionInferenceEngine,
    extract_text_from_pdf,
    extract_cv_bullets,
    detect_bullet_weakness,
    rewrite_bullet,
    rewrite_all_bullets,
)

engine = CVisionInferenceEngine()
STATIC_DIR = ROOT / "static"
FIGURES_DIR = ROOT / "figures"


class CVisionAppHandler(SimpleHTTPRequestHandler):
    """Handles static web requests and REST API inference endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                data = index_path.read_bytes()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

        elif path == "/api/health":
            self._send_json({
                "status": "healthy",
                "framework": "CVision Full-Stack Server",
                "device": "cpu",
                "preprocessors_loaded": engine.preprocessors is not None,
                "dual_branch_net_loaded": engine.dual_net is not None,
                "baseline_loaded": engine.logreg_model is not None or engine.rf_model is not None
            })
            return

        elif path.startswith("/static/"):
            file_path = STATIC_DIR / path.replace("/static/", "")
            if file_path.exists() and file_path.is_file():
                return super().do_GET()

        return super().do_GET()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)

        # ---------------- AI Bullet Rewriter Endpoints ----------------
        if url.path in ("/api/rewrite", "/rewrite"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                self._send_json({"error": "Invalid JSON body provided."}, status=400)
                return

            bullet = payload.get("bullet", "").strip()
            role = payload.get("role", "")
            jd_keywords = payload.get("jd_keywords", [])
            tone = payload.get("tone", "impact")

            if not bullet:
                self._send_json({"error": "Bullet text cannot be empty."}, status=400)
                return
            if len(bullet) > 1000:
                self._send_json({"error": "Bullet text exceeds 1,000 character maximum."}, status=400)
                return

            try:
                res = rewrite_bullet(bullet=bullet, role=role, jd_keywords=jd_keywords, tone=tone)
                self._send_json(res)
            except Exception as exc:
                self._send_json({"error": f"Bullet rewrite failed: {str(exc)}"}, status=500)
            return

        elif url.path in ("/api/rewrite-all", "/rewrite-all"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                self._send_json({"error": "Invalid JSON body provided."}, status=400)
                return

            bullets = payload.get("bullets", [])
            role = payload.get("role", "")
            jd_keywords = payload.get("jd_keywords", [])
            tone = payload.get("tone", "impact")

            if not isinstance(bullets, list) or not bullets:
                self._send_json({"error": "A non-empty list of bullets is required."}, status=400)
                return

            try:
                normalized_bullets = []
                for b in bullets:
                    if isinstance(b, str):
                        normalized_bullets.append({"text": b, "is_weak": True})
                    elif isinstance(b, dict):
                        normalized_bullets.append(dict(b))
                results = rewrite_all_bullets(bullets=normalized_bullets, role=role, jd_keywords=jd_keywords, tone=tone)
                self._send_json({"results": results, "count": len(results), "status": "completed"})
            except Exception as exc:
                self._send_json({"error": f"Batch bullet rewrite failed: {str(exc)}"}, status=500)
            return

        elif url.path in ("/api/extract-bullets", "/extract-bullets"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                self._send_json({"error": "Invalid JSON body provided."}, status=400)
                return

            raw_text = payload.get("text", "")
            bullets = extract_cv_bullets(raw_text)
            for b in bullets:
                b.update(detect_bullet_weakness(b["text"]))
            weak_count = sum(1 for b in bullets if b.get("is_weak"))
            self._send_json({"bullets": bullets, "count": len(bullets), "weakBulletsCount": weak_count})
            return

        elif url.path == "/api/analyze":
            content_length = int(self.headers.get("Content-Length", 0))
            content_type = self.headers.get("Content-Type", "")
            body = self.rfile.read(content_length)

            raw_text = ""
            file_name = "Uploaded_Resume.pdf"
            file_size_str = f"{len(body) // 1024} KB"
            preferred_model = "dual"

            try:
                if "application/json" in content_type:
                    payload = json.loads(body.decode("utf-8"))
                    raw_text = payload.get("text", "")
                    file_name = payload.get("fileName", "Pasted_Resume.txt")
                    preferred_model = payload.get("model", "dual")
                    file_size_str = f"{len(raw_text.encode('utf-8')) // 1024} KB"

                elif "multipart/form-data" in content_type:
                    header_lines = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1")
                    msg = email.parser.BytesParser().parsebytes(header_lines + body)

                    for part in msg.walk():
                        disposition = part.get("Content-Disposition", "")
                        if 'name="model"' in disposition:
                            preferred_model = part.get_payload(decode=True).decode("utf-8").strip()

                        elif 'name="file"' in disposition:
                            raw_filename = part.get_filename() or "Uploaded_Resume.pdf"
                            file_name = raw_filename
                            file_bytes = part.get_payload(decode=True)
                            file_size_str = f"{max(1, len(file_bytes) // 1024)} KB"

                            if raw_filename.lower().endswith(".pdf"):
                                raw_text = extract_text_from_pdf(file_bytes)
                            else:
                                raw_text = file_bytes.decode("utf-8", errors="replace")

                if not raw_text.strip():
                    self._send_json({"error": "No readable text could be extracted from the uploaded file."}, status=400)
                    return

                # Execute real-time inference
                result = engine.predict_resume(
                    raw_text=raw_text,
                    file_name=file_name,
                    file_size_str=file_size_str,
                    preferred_model=preferred_model
                )
                self._send_json(result)

            except Exception as exc:
                self._send_json({"error": f"Inference pipeline failed: {str(exc)}"}, status=500)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _send_json(self, payload: dict, status: int = 200):
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


def run_server(port: int = 8000):
    for candidate_port in (port, 8080, 8888, 5000):
        try:
            server_address = ("", candidate_port)
            httpd = HTTPServer(server_address, CVisionAppHandler)
            url = f"http://localhost:{candidate_port}"
            print("=" * 65)
            print("   🚀 CVision Full-Stack AI Resume Evaluator")
            print("=" * 65)
            print(f" * Server running at: {url}")
            print(f" * UI Dashboard:      {url}/")
            print(f" * Healthcheck:       {url}/api/health")
            print("=" * 65)
            print(" Press Ctrl+C to stop the server.\n")

            # Automatically launch the web dashboard in default browser
            try:
                webbrowser.open(url)
            except Exception:
                pass

            httpd.serve_forever()
            return
        except OSError:
            continue

    print("Error: Could not bind to any port (8000, 8080, 8888).")


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port_arg)

