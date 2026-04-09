# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TinyTTS is an ultra-lightweight English text-to-speech model with ~1.6M parameters (~3.4 MB ONNX). It runs on CPU-only machines, edge devices, and embedded systems without GPU. Distributed via PyPI and npm, with models hosted on HuggingFace (`backtracking/tiny-tts`).

## Commands

```bash
# Install from source (editable)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# CLI inference
tiny-tts --text "Hello world" --device cpu

# Run Gradio web demo
python app.py

# Benchmark against other TTS engines
python benchmark.py        # PyTorch + ONNX vs Piper/Kokoro/KittenTTS
python benchmark_onnx.py   # ONNX-only

# Export PyTorch model to ONNX (requires G.pth checkpoint)
python export_onnx.py

# Lint & format
black tiny_tts/
flake8 tiny_tts/

# Tests
pytest
```

## Architecture

### TTS Pipeline

```
Text → normalize_text() → grapheme_to_phoneme() → phonemes_to_ids() → blank insertion →
VoiceSynthesizer (encoder → duration predictor → flow → decoder) → 44.1kHz WAV
```

### Key Modules

- **`tiny_tts/infer.py`** — PyTorch inference engine and CLI entry point (`tiny-tts` command). Auto-downloads checkpoint from HuggingFace if missing.
- **`tiny_tts/infer_onnx.py`** — ONNX Runtime inference. Uses 4 separate ONNX submodels: `text_encoder`, `duration_predictor`, `flow`, `decoder`.
- **`tiny_tts/models/synthesizer.py`** — `VoiceSynthesizer` model definition. Encoder with speaker conditioning, noise-scaled monotonic alignment search (MAS), HiFi-GAN-style decoder.
- **`tiny_tts/text/english.py`** — Text normalization and G2P pipeline. Uses BERT tokenizer + `g2p_en` neural fallback + CMU Pronouncing Dictionary (123K entries in `cmudict.rep`).
- **`tiny_tts/text/symbols.py`** — 39 English phone symbols and mapping tables.
- **`tiny_tts/nn/`** — Neural network building blocks: attention (`attentions.py`), convolution modules (`modules.py`), flow transforms (`transforms.py`), utilities (`commons.py`).
- **`tiny_tts/alignment/core.py`** — Viterbi alignment with Numba JIT compilation.
- **`tiny_tts/utils/config.py`** — All hyperparameters: sample rate (44100), model dimensions (inter_channels=32, hidden_channels=32), upsample rates, speaker/language IDs.
- **`tiny_tts/__init__.py`** — Public API: `TinyTTS` class with `speak()` method.

### Public API

```python
from tiny_tts import TinyTTS
tts = TinyTTS()  # auto-detects device, downloads checkpoint
tts.speak("text", output_path="out.wav", speaker="MALE", speed=1.0)
```

### Speed Control

Speed is implemented via `length_scale` parameter (inverse of speed value). The `speed` CLI arg and API param are converted: `length_scale = 1.0 / speed`.

### ONNX Export

The model is split into 4 ONNX submodules for optimized inference. ONNX provides ~3x speedup over PyTorch on CPU (~53x real-time).

### Dual Distribution

Python package uses PyTorch or ONNX. Node.js package (`npm`) uses pure ONNX with a JS-ported `g2p_en` neural model — zero Python dependency, 100% phoneme match with Python.

## Important Notes

- Model checkpoints (`*.pth`, `*.onnx`) are not in the repo — auto-downloaded from HuggingFace Hub on first use.
- Inference outputs go to `infer_outputs/` directory.
- Currently English-only, single speaker (MALE). Architecture supports multilingual via tone/language token embeddings but is not trained for it.
- BERT features in the model are disabled (zeroed out) — the model works without them.
- Training code is not released.
