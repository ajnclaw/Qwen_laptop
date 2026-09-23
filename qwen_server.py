# qwen_server.py
#
# Persistent HTTP server exposing Qwen-Image (text2img) and Qwen-Image-Edit
# (img2img) via WeeLLM's layer-streaming pipelines. Keeps both pipelines
# loaded in memory between requests -- the ~13min/image cost we measured
# includes a full pipeline load; a persistent server pays that once, not
# per image.
#
# Mirrors main.py's own call pattern exactly (introspecting pipe.__call__
# to handle Qwen's guidance_scale -> true_cfg_scale rename and to strip
# kwargs the specific pipeline doesn't accept) rather than the simplified
# .generate() wrapper shown in the README, since main.py is the version
# we've actually confirmed works end to end.
#
# Run with: .venv\Scripts\python.exe qwen_server.py

import base64
import inspect
import io
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

from weellm import WeeImageToImagePipeline, WeeTextToImagePipeline


HOST = "0.0.0.0"
PORT = 8421

TEXT2IMG_MODEL = "Qwen/Qwen-Image"
EDIT_MODEL = "Qwen/Qwen-Image-Edit"

NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, cgi, render, plastic skin, "
    "over-saturated, deformed, blurry, low quality, uncanny"
)

_t2i_pipe = None
_edit_pipe = None


def _load_t2i():
    global _t2i_pipe
    if _t2i_pipe is None:
        print("[qwen_server] Loading text2img pipeline (first call only)...")
        _t2i_pipe = WeeTextToImagePipeline.from_pretrained(
            TEXT2IMG_MODEL, device="cuda", torch_dtype=torch.bfloat16
        )
    return _t2i_pipe


def _load_edit():
    global _edit_pipe
    if _edit_pipe is None:
        print("[qwen_server] Loading image-edit pipeline (first call only)...")
        _edit_pipe = WeeImageToImagePipeline.from_pretrained(
            EDIT_MODEL, device="cuda", torch_dtype=torch.bfloat16
        )
    return _edit_pipe


def _call_pipe(pipe, **call_kwargs):
    """
    Same introspection main.py uses before calling pipe(**call_kwargs) --
    different underlying diffusers pipelines (Qwen vs SDXL vs Flux) expect
    different exact kwarg names, so this adapts rather than hardcoding one.
    """
    sig = inspect.signature(pipe.__call__)
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    expected = set(sig.parameters.keys())

    if "true_cfg_scale" in expected and "guidance_scale" in call_kwargs:
        call_kwargs["true_cfg_scale"] = call_kwargs.pop("guidance_scale")

    if not has_kwargs:
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in expected}

    return pipe(**call_kwargs)


class Handler(BaseHTTPRequestHandler):
    def _send_image(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(buffer.tell()))
        self.end_headers()
        self.wfile.write(buffer.getvalue())

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
            seed = body.get("seed", 42)
            start = time.time()

            if self.path == "/generate":
                pipe = _load_t2i()
                generator = torch.Generator(device=pipe.device).manual_seed(seed)

                out = _call_pipe(
                    pipe,
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=body.get("height", 1024),
                    width=body.get("width", 576),
                    num_inference_steps=body.get("steps", 10),
                    guidance_scale=body.get("guidance_scale", 3.5),
                    generator=generator,
                )
                self._send_image(out.images[0])

            elif self.path == "/edit":
                pipe = _load_edit()
                generator = torch.Generator(device=pipe.device).manual_seed(seed)
                init_image = Image.open(
                    io.BytesIO(base64.b64decode(body["image_b64"]))
                ).convert("RGB")

                out = _call_pipe(
                    pipe,
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=init_image,
                    strength=body.get("strength", 0.6),
                    height=body.get("height", 1024),
                    width=body.get("width", 576),
                    num_inference_steps=body.get("steps", 20),
                    guidance_scale=body.get("guidance_scale", 3.5),
                    generator=generator,
                )
                self._send_image(out.images[0])

            else:
                self.send_error(404, "Not found")
                return

            print(f"[qwen_server] {self.path} done in {time.time() - start:.1f}s")

        except Exception:
            traceback.print_exc()
            self._send_error(traceback.format_exc())

    def log_message(self, format, *args):
        print(f"[qwen_server] {self.address_string()} - {format % args}")


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Qwen image server listening on {HOST}:{PORT}")
    print("Both pipelines load lazily on first use -- first /generate or")
    print("/edit call will be slow (model load), later calls reuse it.")
    server.serve_forever()


if __name__ == "__main__":
    main()
