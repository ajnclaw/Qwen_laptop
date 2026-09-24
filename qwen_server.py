# qwen_server.py
#
# HTTP server exposing Qwen-Image (text2img) via WeeLLM, one fresh
# subprocess per request rather than a persistent in-memory pipeline.
#
# We tried keeping the pipeline warm across requests to save reload
# time, but every single successful generation all night -- on WSL2
# and on native Windows -- went through main.py's one-shot CLI, which
# loads fresh and exits. Every failure went through the persistent
# version, hitting "Cannot copy out of meta tensor; no data!" even on
# requests with no prior interruption -- something in WeeLLM's layer
# eviction/reload bookkeeping doesn't survive being reused across
# multiple requests in one long-lived process. This trades away the
# reload-time saving for the only pattern that's actually proven
# reliable: exactly mirroring the validated main.py CLI invocation,
# fresh process, every time.
#
# Run with: .venv\Scripts\python.exe qwen_server.py

import json
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "0.0.0.0"
PORT = 8421

TEXT2IMG_MODEL = "Qwen/Qwen-Image"
EDIT_MODEL = "Qwen/Qwen-Image-Edit"

NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, cgi, render, plastic skin, "
    "over-saturated, deformed, blurry, low quality, uncanny"
)

MAIN_PY = Path(__file__).parent / "main.py"

# Only one generation at a time -- these are subprocesses on a single
# 6GB GPU, running two at once would fight over VRAM.
_generation_lock = threading.Lock()


def _run_generation(prompt, output_path, height, width, steps, guidance_scale, seed,
                     image_path=None, strength=None):
    cmd = [
        sys.executable, str(MAIN_PY),
        "--model", EDIT_MODEL if image_path else TEXT2IMG_MODEL,
        "--prompt", prompt,
        "--negative_prompt", NEGATIVE_PROMPT,
        "--height", str(height),
        "--width", str(width),
        "--steps", str(steps),
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed),
        "--dtype", "bfloat16",
        "--output", str(output_path),
        "--verbose",
    ]

    if image_path:
        cmd += ["--image", str(image_path)]
    if strength is not None:
        cmd += ["--strength", str(strength)]

    result = subprocess.run(
        cmd,
        cwd=str(MAIN_PY.parent),
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"main.py exited {result.returncode}\n"
            f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
        )

    return output_path.read_bytes()


class Handler(BaseHTTPRequestHandler):
    def _send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, message):
        self.send_response(500)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            prompt = body["prompt"]
            start = time.time()

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "out.png"
                image_path = None

                if self.path == "/edit":
                    import base64

                    image_path = Path(tmpdir) / "ref.png"
                    image_path.write_bytes(base64.b64decode(body["image_b64"]))
                elif self.path != "/generate":
                    self.send_error(404, "Not found")
                    return

                with _generation_lock:
                    data = _run_generation(
                        prompt=prompt,
                        output_path=output_path,
                        height=body.get("height", 1024),
                        width=body.get("width", 576),
                        steps=body.get("steps", 20 if image_path else 10),
                        guidance_scale=body.get("guidance_scale", 3.5),
                        seed=body.get("seed", 42),
                        image_path=image_path,
                        strength=body.get("strength") if image_path else None,
                    )

                self._send_bytes(data, "image/png")

            print(f"[qwen_server] {self.path} done in {time.time() - start:.1f}s")

        except Exception:
            traceback.print_exc()
            self._send_error(traceback.format_exc())

    def log_message(self, format, *args):
        print(f"[qwen_server] {self.address_string()} - {format % args}")


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Qwen image server listening on {HOST}:{PORT}")
    print("Each request spawns a fresh main.py subprocess (no warm model")
    print("kept in memory) -- slower per-call, but this is the only")
    print("pattern that's proven reliable so far.")
    server.serve_forever()


if __name__ == "__main__":
    main()
