# 🫁 EchoGlow — Neonatal ICU Distress Analyzer

> **AI-powered neonatal monitoring that runs on a $130 edge device, fully offline.**
> Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) · Powered by **Gemma 4 E4B** via Ollama

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Gemma 4](https://img.shields.io/badge/Gemma_4-E4B-teal?logo=google)](https://ollama.com/library/gemma4)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## The Problem

In understaffed public hospitals and rural clinics, neonatal distress is often caught too late. A single nurse may monitor six or more infants simultaneously. Commercial multi-parameter monitors cost $8,000–$15,000 — unaffordable for the clinics that need them most.

Infant distress is primarily acoustic. Pain cries, hunger cries, stridor, and wheezing each have clinically distinct spectral patterns — but recognising them in a noisy ward at 3 AM is beyond human capacity at scale.

## The Solution

EchoGlow is a **local-first, multimodal neonatal monitoring assistant** that:

- 🎤 **Continuously analyses audio** from a microphone near the crib, extracting pitch, energy, spectral features, and cry patterns
- 📷 **Optionally analyses camera frames** using Gemma 4's native vision capability
- 🤖 **Runs Gemma 4 E4B locally** via Ollama — no cloud, no internet required, zero data egress
- 🚨 **Triggers graded alerts** (Monitor / Urgent / Critical) with audio alarms and clinical recommendations
- 📄 **Generates PDF clinical reports** for patient records
- 💰 **Runs on $130 hardware** (Raspberry Pi 5 + USB microphone + webcam)

---

## Demo

[![EchoGlow Demo](https://img.shields.io/badge/▶_Watch_Demo-YouTube-red?logo=youtube)](YOUR_YOUTUBE_LINK_HERE)

Press `🎬 Full Demo` in the app to run the complete 2-minute scripted demonstration automatically.

![EchoGlow Dashboard](docs/screenshot.png)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Edge Device (100% local)                  │
│                                                              │
│  USB Mic ──► AudioProcessor ──► AudioFeatures (numpy DSP)   │
│                                       │                      │
│  Camera ──► VisionProcessor ──► JPEG frame (OpenCV)         │
│                                       │                      │
│                              GemmaAnalyzer                   │
│                         ┌─────────────────────┐             │
│                         │   Gemma 4 E4B        │             │
│                         │   via Ollama API     │             │
│                         │   /api/chat          │             │
│                         │   format: json       │             │
│                         └──────────┬──────────┘             │
│                                    │                         │
│                         JSON Clinical Assessment             │
│                    {classification, severity,                │
│                     confidence, reasoning, action}           │
│                                    │                         │
│            FastAPI Server + WebSocket Broadcast              │
│                                    │                         │
│            Browser Dashboard (HTML + Canvas)                 │
│            ├─ Rolling waveform (12 fps heartbeat)            │
│            ├─ Classification badge (8 categories)            │
│            ├─ Clinical reasoning text                        │
│            ├─ Real-time alert feed                           │
│            └─ PDF report generator                           │
│                                                              │
│   ┌─────────────────────────────────────────────┐           │
│   │  PATIENT DATA NEVER LEAVES THIS DEVICE       │           │
│   └─────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

### Analysis Modes

| Mode | Audio | Camera | Use Case |
|---|---|---|---|
| 🎤 Audio | ✅ Full DSP | ❌ | Acoustic-only monitoring |
| 📷 Video | Context only | ✅ Camera/upload | Visual distress assessment |
| 🎤+📷 Both | ✅ Full DSP | ✅ Camera/upload | Combined multimodal |

### Why Gemma 4 E4B?

| Requirement | How E4B Meets It |
|---|---|
| Edge deployment | Fits in 4 GB VRAM — runs on Raspberry Pi 5 |
| Multimodal | Native text + image in one inference call |
| Clinical reasoning | Returns structured JSON with human-readable explanations |
| Privacy | On-device only — open weights, no API key |
| Cost | Free to run after hardware — no cloud subscription |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed
- Gemma 4 model pulled

```bash
# Pull the model (~3.5 GB download)
ollama pull gemma4:4b

# Start Ollama (keep this terminal open)
ollama serve
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/echoglow
cd echoglow
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open **http://localhost:8765** in your browser.

- Click **▶ Start** for live microphone monitoring
- Click **🎬 Full Demo** for the automated submission demo sequence
- Click **Demo ↔ Live** toggle to switch between simulation and real microphone

---

## Project Structure

```
echoglow/
├── app.py               # FastAPI server, WebSocket streaming, REST API
├── audio_processor.py   # Microphone capture, DSP feature extraction, simulation
├── gemma_analyzer.py    # Gemma 4 via Ollama — prompts, JSON parsing, fallback
├── vision_processor.py  # Optional camera frame capture (OpenCV)
├── report_generator.py  # PDF clinical report generation (fpdf2)
├── config.py            # Configuration (model, ports, thresholds)
├── requirements.txt
└── static/
    └── index.html       # Single-file dashboard (waveform, analysis, alerts)
```

### Key API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `WS` | `/ws` | Real-time analysis stream |
| `POST` | `/api/start` | Begin session (`simulation`, `analysis_mode`) |
| `POST` | `/api/stop` | End session |
| `POST` | `/api/pause` | Pause Gemma inference (waveform continues) |
| `POST` | `/api/resume` | Resume inference |
| `POST` | `/api/set-mode` | Switch analysis mode mid-session |
| `POST` | `/api/analyze-image` | One-shot image analysis |
| `GET` | `/api/snapshot` | Latest camera frame |
| `GET` | `/api/report?hours=N` | Download PDF report |

---

## Gemma 4 Integration Detail

EchoGlow uses Gemma 4 in three ways:

**1. Audio classification (primary)**
Structured audio features are described in natural language and passed to Gemma 4 as a clinical sensor report:
```
=== EchoGlow Sensor Report — 14:22:07 ===
Cry detected     : YES — ongoing for 8.3 s
Energy (RMS)     : 0.7200  [HIGH]
Pitch estimate   : 510.0 Hz
Intensity pattern: escalating
Audio observation:
High-pitched escalating cry at 510 Hz. High intensity. Pain pattern.
```

**2. Visual analysis (optional)**
Camera frames are passed alongside the audio description to Gemma 4's vision input. The model combines acoustic and visual cues — skin colour, chest movement, posture — in a single inference call.

**3. Structured output enforcement**
Ollama's `format: json` parameter combined with a strict system prompt ensures consistent machine-parseable output:
```json
{
  "cry_classification": "pain",
  "severity": "urgent",
  "confidence": 0.85,
  "reasoning": "High-pitched escalating cry at 510 Hz with urgent pattern...",
  "clinical_note": "Acute pain cry detected. Assess infant within 2 minutes.",
  "action_recommended": "Check IV site, positioning, recent procedures.",
  "respiratory_concern": false
}
```

---

## Clinical Classifications

| Classification | Severity | Audio Signature |
|---|---|---|
| Calm | 🟢 Normal | No vocalization |
| Hunger | 🟡 Monitor | Rhythmic, 300–420 Hz, periodic pauses |
| Discomfort | 🟡 Monitor | Fussy, irregular, low-moderate |
| Pain | 🔴 Urgent | High-pitched >450 Hz, sudden, escalating |
| Stridor | 🔴 Critical | Harsh inspiratory noise, possible airway obstruction |
| Wheezing | 🔴 Critical | Expiratory musical sound, lower airway |
| Respiratory Distress | 🔴 Critical | Any abnormal breathing pattern |

---

## Hardware Bill of Materials (Full Deployment)

| Component | Cost |
|---|---|
| Raspberry Pi 5 (8GB) | ~$80 |
| USB Microphone | ~$15 |
| USB Webcam | ~$25 |
| MicroSD card (32GB) | ~$8 |
| **Total** | **~$128** |

Compared to: $8,000–$15,000 for commercial NICU monitors.

---

## Configuration

Edit `config.py` or set environment variables:

```python
OLLAMA_URL  = "http://localhost:11434"   # Ollama API base
GEMMA_MODEL = "gemma4:4b"               # Model name (check with `ollama list`)
ANALYSIS_INTERVAL = 7.0                 # Seconds between Gemma calls
SAMPLE_RATE = 16000                     # Audio capture rate (Hz)
PORT = 8765                             # Dashboard port
```

---

## Disclaimer

EchoGlow is a research prototype and AI-assisted monitoring aid. It does **not** replace qualified clinical judgment. All alerts must be evaluated by competent medical personnel. Not approved for clinical use.

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Acknowledgements

- [Google DeepMind](https://deepmind.google/) for Gemma 4 open weights
- [Ollama](https://ollama.com) for local LLM serving
- [FastAPI](https://fastapi.tiangolo.com) for the async backend
- Built for the **Gemma 4 Good Hackathon** — *"AI for Good"*
