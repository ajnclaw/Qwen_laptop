# Running on the Windows image-generation laptop

This is a fork of [Jit-Roy/WeeLLM](https://github.com/Jit-Roy/WeeLLM) with one
fix applied: `weellm/pipelines/weebasepipeline.py`'s `defrag_vae_decode` would
infinite-recurse for any VAE (including `Qwen/Qwen-Image`'s) that doesn't
override `decode` itself, since `BaseVAEStreamer.__getattr__` proxies
unresolved attributes straight back to the same object being monkey-patched.
See the diff in `weellm/pipelines/weebasepipeline.py` for the fix.

Kept as a **separate repo/venv from the image-server project** on purpose --
this pulls a different `diffusers` build (`git+https://github.com/huggingface/diffusers`,
the dev branch) that would otherwise conflict with the pinned release version
the production image server depends on.

## Setup (first time only)

```powershell
git clone <this repo's URL>
cd Qwen_laptop
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
```

Set an `HF_TOKEN` environment variable first if you haven't already elsewhere
on this machine -- Qwen-Image is a large download (~14GB+) and unauthenticated
Hugging Face requests are heavily rate-limited:

```powershell
$env:HF_TOKEN = "hf_..."
```

## Running

```powershell
.venv\Scripts\python.exe main.py --model "Qwen/Qwen-Image" --prompt "your prompt here" --negative_prompt "cartoon, illustration, painting, cgi, render, plastic skin, over-saturated, deformed, blurry, low quality, uncanny" --height 1024 --width 576 --steps 10 --guidance_scale 3.5 --seed 42 --dtype bfloat16 --output "output.png" --verbose
```

Note `--vram_budget`/`--ram_budget` are **omitted** deliberately -- leaving
them unset lets WeeLLM auto-detect actual available hardware ("Dynamic" mode)
instead of being pinned to a manually guessed number. Pass them explicitly
only if you want to force a specific ceiling.

## What we already learned testing this on WSL2

- The recursion bug above blocked every model using the generic `AutoencoderKL`
  VAE streaming path, not just Qwen-Image specifically.
- On WSL2, one Qwen-Image generation (10 steps) took ~78 minutes despite peak
  VRAM staying at ~2.1GB, well under budget. That's almost certainly WSL2's
  virtualized filesystem, not the technique itself -- WeeLLM's own README
  notes it's optimized for direct NVMe access, which WSL2 doesn't give it.
  This machine runs native Windows specifically to test that theory.
- Quality result was genuinely good: correct octopus anatomy (8 arms, correct
  mantle shape) on the first real run -- better than SD1.5+LCM-LoRA managed
  on the same prompt.
