#!/usr/bin/env python3
import json
import mimetypes
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent / "dist"
UPSTREAM = os.environ.get("GLM_UPSTREAM", "http://127.0.0.1:8017")
HOST = os.environ.get("GLM_UI_HOST", "127.0.0.1")
PORT = int(os.environ.get("GLM_UI_PORT", "8018"))


class Handler(BaseHTTPRequestHandler):
    server_version = "GLMTestUI/1.0"

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def proxy(self, method, path):
        body = None
        if method == "POST":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
        req = urllib.request.Request(
            UPSTREAM + path,
            data=body,
            method=method,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                payload = response.read()
                self.send_bytes(response.status, payload, response.headers.get_content_type())
        except urllib.error.HTTPError as exc:
            self.send_bytes(exc.code, exc.read(), exc.headers.get_content_type())
        except Exception as exc:
            payload = json.dumps({"error": {"message": str(exc)}}).encode()
            self.send_bytes(502, payload, "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self.proxy("GET", "/health")
        if path == "/api/models":
            return self.proxy("GET", "/v1/models")
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (ROOT / relative).resolve()
        if ROOT.resolve() not in target.parents and target != ROOT.resolve():
            return self.send_bytes(403, b"Forbidden", "text/plain")
        if not target.is_file():
            target = ROOT / "index.html"
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_bytes(200, target.read_bytes(), mime)

    def do_POST(self):
        if urlparse(self.path).path == "/api/chat":
            return self.proxy("POST", "/v1/chat/completions")
        self.send_bytes(404, b"Not found", "text/plain")


if __name__ == "__main__":
    print(f"GLM Test UI: http://{HOST}:{PORT} -> {UPSTREAM}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
