"""Servidor local do painel com suporte a Range (necessário para saltar no vídeo de 1,7 GB).
Uso: .venv/bin/python painel/serve.py [porta]
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_POST(self):
        import json
        if self.path.startswith("/api/shot"):   # guarda uma imagem de verificação (QA) enviada pelo painel
            import urllib.parse
            name = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("name", ["shot"])[0]
            n = int(self.headers.get("Content-Length", 0))
            os.makedirs(os.path.join(ROOT, "qa"), exist_ok=True)
            open(os.path.join(ROOT, "qa", os.path.basename(name) + ".jpg"), "wb").write(self.rfile.read(n))
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
            return
        if self.path != "/api/label":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        path = os.path.join(ROOT, "labels.json")
        cur = json.load(open(path)) if os.path.exists(path) else {}
        if d.get("value"):
            cur[d["key"]] = d["value"]
        else:
            cur.pop(d["key"], None)
        json.dump(cur, open(path, "w"), ensure_ascii=False, indent=1)
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        rem = getattr(self, "_remaining", None)
        try:
            if rem is None:
                return super().copyfile(source, outputfile)
            while rem > 0:
                buf = source.read(min(1 << 20, rem))
                if not buf:
                    break
                outputfile.write(buf)
                rem -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self._remaining = None


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8780
    print(f"Painel em http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
