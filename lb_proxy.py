"""Tiny round-robin HTTP proxy: :8020 -> [:8013, :8014] (two TP=4 step100 serves).
Per-request round-robin (chat/completions are stateless, full context each call, so no
affinity needed). Lets one harbor eval (single api_base) drive both serves = 2x GPUs."""
import http.server, socketserver, urllib.request, urllib.error, itertools, threading
BACKENDS = ["http://localhost:8013", "http://localhost:8014"]
_rr = itertools.cycle(BACKENDS); _lock = threading.Lock()
def _next():
    with _lock: return next(_rr)

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _proxy(self, method):
        b = _next()
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else None
        req = urllib.request.Request(b + self.path, data=body, method=method)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length"): req.add_header(k, v)
        try:
            r = urllib.request.urlopen(req, timeout=1800)
            data = r.read(); code = r.status
            self.send_response(code)
            self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read(); self.send_response(e.code)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        except Exception as e:
            msg = str(e).encode()[:200]; self.send_response(502)
            self.send_header("Content-Length", str(len(msg))); self.end_headers(); self.wfile.write(msg)
    def do_POST(self): self._proxy("POST")
    def do_GET(self): self._proxy("GET")
    def log_message(self, *a): pass

class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
TS(("127.0.0.1", 8020), H).serve_forever()
