<div align="center">
<img src="docs/thumbnail.jpg" alt="EchoGlow banner" width="100%"/>
🫁 EchoGlow
Neonatal ICU Distress Analyzer
AI-powered infant monitoring that runs on a $130 edge device — fully offline, fully private.
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Gemma 4 E4B](https://img.shields.io/badge/Gemma_4-E4B-00C4B4?logo=google&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local_inference-black?logo=ollama)
![License: MIT](https://img.shields.io/badge/License-MIT-F5A623)
![Kaggle](https://img.shields.io/badge/Kaggle-Gemma_4_Good_Hackathon-20BEFF?logo=kaggle)

</div>
---
The Problem
Every year, 2.3 million newborns die within their first month of life. A significant proportion of these deaths are preventable — not for lack of medicine, but for lack of timely detection.
In understaffed public hospitals and rural clinics across low- and middle-income countries, a single nurse may monitor six or more infants simultaneously. Commercial multi-parameter neonatal monitors cost $8,000–$15,000 — unaffordable for the clinics that need them most.
Infant distress is primarily acoustic. A pain cry sounds fundamentally different from a hunger cry. A stridor — the harsh inspiratory wheeze of a partially obstructed airway — is clinically distinct from a wheezing expiration. But recognising these differences in a noisy ward, at 3 AM, across six cribs, is beyond human capacity at scale.
EchoGlow asks: what if the AI model already on your phone could close that gap?
---
The Solution
EchoGlow is a local-first, multimodal neonatal monitoring assistant powered by Gemma 4 E4B. It continuously analyses audio from a microphone near an infant's crib, optionally pairs this with a camera feed, and classifies distress signals in real time — all on-device, all offline.
Feature	Description
🎤 Audio Analysis	Real-time cry pattern detection via numpy DSP + Gemma 4 reasoning
📷 Visual Analysis	Camera frame analysis using Gemma 4's native vision input
🎤+📷 Combined Mode	Audio features + camera frame in a single Gemma 4 inference call
🚨 Graded Alerts	Monitor / Urgent / Critical with audio alarms and screen flash
⏸ Pause & Resume	Freeze analysis during image review; waveform continues
📄 PDF Reports	Auto-generated clinical summaries (1h / 7h / 12h / 24h windows)
🎬 Full Demo Mode	Scripted 2:20 demonstration — no hardware required
🔒 Zero Data Egress	Patient audio and images never leave the device
---
Demo
> Press **🎬 Full Demo** in the app to run the complete scripted 2-minute demonstration automatically — no microphone or camera required.
The demo cycles through: baseline calm → hunger cry → pain cry (URGENT alert) → stridor (CRITICAL alert) → visual camera analysis → combined audio+visual → PDF report download.

---
Table of Contents
Architecture
Why Gemma 4 E4B?
Clinical Classifications
Quick Start
Prerequisites
Windows
Linux / macOS
Raspberry Pi
Configuration
Project Structure
API Reference
Gemma 4 Integration
Hardware BOM
Troubleshooting
Roadmap
Disclaimer
License
---
Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     Edge Device  (100% local)                   │
│                                                                  │
│  USB Microphone ──► audio_processor.py                          │
│                      numpy DSP pipeline:                        │
│                      · RMS energy    · Pitch estimate (Hz)      │
│                      · Spectral centroid / ZCR                  │
│                      · Cry duration  · Intensity pattern        │
│                      · 9-scenario simulation (demo mode)        │
│                              │                                  │
│  USB Camera ────► vision_processor.py                           │
│                      OpenCV capture → 320×240 JPEG              │
│                      Base64 encoded, on-demand                  │
│                              │                                  │
│                              ▼                                  │
│                   gemma_analyzer.py                             │
│              ┌──────────────────────────┐                       │
│              │     Gemma 4 E4B          │                       │
│              │  via Ollama /api/chat    │                       │
│              │  format: json            │                       │
│              │  4B params · 4 GB VRAM  │                       │
│              └────────────┬─────────────┘                       │
│                           │                                     │
│              JSON clinical assessment:                          │
│              { cry_classification, severity,                    │
│                confidence, reasoning,                           │
│                clinical_note, action_recommended,               │
│                respiratory_concern }                            │
│                           │                                     │
│          app.py — FastAPI + WebSocket server                    │
│                           │                                     │
│          static/index.html — Browser dashboard                  │
│          ├─ Rolling waveform canvas  (12 fps)                   │
│          ├─ Classification badge + severity ring                │
│          ├─ Gemma 4 clinical reasoning text                     │
│          ├─ Real-time graded alert feed                         │
│          ├─ Camera preview / photo upload                       │
│          └─ report_generator.py → PDF download                  │
│                                                                  │
│  ╔══════════════════════════════════════════════════════╗       │
│  ║  PATIENT DATA NEVER LEAVES THIS DEVICE               ║       │
│  ╚══════════════════════════════════════════════════════╝       │
└─────────────────────────────────────────────────────────────────┘
```
Analysis Modes
Mode	Audio sent to Gemma 4	Camera / Photo	Best for
🎤 Audio	Full DSP feature vector	—	Acoustic-only monitoring
📷 Video	Minimal context	✅ Camera frame or upload	Visual-only assessment
🎤+📷 Both	Full DSP feature vector	✅ Camera frame or upload	Richest combined analysis
Modes are switchable live during a session — no restart required.
---
Why Gemma 4 E4B?
The model choice is the core architectural decision of EchoGlow, and it is intentional.
Requirement	How Gemma 4 E4B meets it
Edge deployment	Fits in 4 GB VRAM. Runs on Raspberry Pi 5, mid-range laptop, or Android NPU
Multimodal	Native text + image in a single inference call — no separate vision pipeline
Clinical reasoning	Explains its conclusions in human-readable language a nurse can act on
Structured output	Reliable JSON via `format: json` + strict system prompt + three-pass parser
Privacy	Open weights, runs via Ollama — no API key, no cloud call
Cost	Free to run after hardware — no subscription, no per-inference charge
The 26B MoE and 31B Dense variants would achieve higher accuracy but cannot run on a Raspberry Pi — making the mission-critical $130 deployment target impossible. E4B is not a compromise; it is the correct model for this use case.
---
Clinical Classifications
EchoGlow classifies eight infant states, each mapped to a severity level and a recommended nurse action:
Classification	Severity	Audio signature	Action
Calm	🟢 Normal	No vocalization	Continue routine monitoring
Hunger	🟡 Monitor	Rhythmic 300–420 Hz, periodic pauses	Check last feed time
Discomfort	🟡 Monitor	Fussy, irregular, low-moderate intensity	Assess position, temperature, nappy
Pain	🔴 Urgent	High-pitched >450 Hz, sudden escalating onset	Assess infant within 2 minutes
Stridor	🔴 Critical	Harsh inspiratory noise, ~285 Hz	Immediate airway assessment
Wheezing	🔴 Critical	Expiratory musical sound, lower frequency	Immediate respiratory assessment
Respiratory Distress	🔴 Critical	Any abnormal breathing pattern	Emergency response
Unknown	—	Insufficient signal	Continue monitoring, increase sensitivity
---
Quick Start
Prerequisites
Python 3.10+
Ollama installed and running
Gemma 4 model pulled (≈3.5 GB download)
```bash
# Verify Ollama is installed
ollama --version

# Pull the model
ollama pull gemma4:4b

# Start Ollama server (keep this terminal open)
ollama serve

# Confirm model is ready
ollama list
# Should show: gemma4:4b   ...   3.8 GB
```
> **Download interrupted?** Ollama resumes from where it stopped — just run `ollama pull gemma4:4b` again.
---
Windows Setup
```powershell
# 1. Clone the repository
git clone https://github.com/quartzap/EchoGlow.git
cd EchoGlow

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```
Open http://localhost:8765 in your browser.
> **PortAudio missing?** If `sounddevice` fails to install, run:
> ```powershell
> pip install pipwin
> pipwin install pyaudio
> pip install sounddevice
> ```
> Or simply use **Demo Mode** which requires no microphone.
---
Linux / macOS Setup
```bash
# Install PortAudio (required for live microphone input)
# Ubuntu/Debian:
sudo apt install portaudio19-dev python3-dev

# macOS:
brew install portaudio

# Clone and install
git clone https://github.com/quartzap/EchoGlow.git
cd EchoGlow
pip install -r requirements.txt

# Run
python app.py
```
---
Raspberry Pi Setup
EchoGlow is designed to run on Raspberry Pi 5 (8GB recommended).
```bash
# Install system dependencies
sudo apt update
sudo apt install -y portaudio19-dev python3-dev python3-pip \
                    libopencv-dev python3-opencv git

# Install Ollama for ARM
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model (E2B is lighter for Pi 4, E4B for Pi 5)
ollama pull gemma4:4b   # Pi 5 with 8GB RAM
# or
ollama pull gemma4:2b   # Pi 4 or Pi 5 with 4GB RAM

# Clone and install EchoGlow
git clone https://github.com/quartzap/EchoGlow.git
cd EchoGlow
pip install -r requirements.txt

# Run
python app.py
```
Access the dashboard from any device on the same network:
```
http://<raspberry-pi-ip>:8765
```
> Set `HOST = "0.0.0.0"` in `config.py` to listen on all interfaces.
---
Using the Dashboard
Action	How
Start monitoring	Click ▶ Start (live mic) or Demo Mode ↔ Live toggle
Run scripted demo	Click 🎬 Full Demo
Switch analysis mode	Click 🎤 Audio / 📷 Video / 🎤+📷 Both selector
Pause analysis	Click ⏸ Pause (waveform continues)
Analyse a photo	Upload image when no camera detected
Download report	Click 📥 Report ▾ → select time window
Toggle sound alerts	Click 🔔 Alerts: ON/OFF
---
Configuration
Edit `config.py` or set environment variables before running:
```python
# config.py

# ── Ollama / Gemma ──────────────────────────────────────────
OLLAMA_URL  = os.getenv("OLLAMA_URL",  "http://localhost:11434")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma4:4b")
# Use "gemma4:2b" for E2B (lighter, faster, Raspberry Pi 4)

# ── Audio ────────────────────────────────────────────────────
SAMPLE_RATE       = 16000   # Hz — microphone capture rate
WINDOW_SIZE       = 3.0     # seconds of audio in circular buffer
ANALYSIS_INTERVAL = 7.0     # seconds between Gemma 4 calls

# ── Server ───────────────────────────────────────────────────
HOST = "0.0.0.0"   # "0.0.0.0" to expose on network, "127.0.0.1" for local only
PORT = 8765
```
Environment variable overrides (useful for deployment):
```bash
OLLAMA_URL=http://192.168.1.50:11434 GEMMA_MODEL=gemma4:2b python app.py
```
---
Project Structure
```
EchoGlow/
│
├── app.py                  # FastAPI server, WebSocket broadcast, REST endpoints
├── audio_processor.py      # Microphone capture (sounddevice), numpy DSP, simulation
├── gemma_analyzer.py       # Gemma 4 via Ollama — prompts, JSON parsing, fallback
├── vision_processor.py     # Camera frame capture (OpenCV), base64 encoding
├── report_generator.py     # PDF clinical reports (fpdf2)
├── config.py               # All tunable constants
├── requirements.txt
│
├── static/
│   └── index.html          # Complete single-file dashboard
│                           # (waveform, analysis panel, alert feed, demo runner)
│
└── docs/
    ├── thumbnail.jpg       # Repository cover image
    ├── architecture.svg    # System architecture diagram
    └── screenshot.png      # Dashboard screenshot
```
---
API Reference
WebSocket — `/ws`
Connects and receives a continuous stream of JSON messages:
Message type	Fields	Description
`state`	`session`, `ollama_ok`, `mic_ok`, `camera_ok`, `model`	Initial state on connect
`heartbeat`	`rms`, `cry_detected`, `pitch`, `waveform_point`, `session_duration`	Waveform update (~12 fps)
`analysis`	`features`, `analysis`, `waveform`, `alert_count`, `frame`	Full Gemma 4 result (every 7s)
`image_analysis`	`analysis`, `alert_count`, `timestamp`	Photo upload result
`paused` / `resumed`	`timestamp`	Session state change
`mode_changed`	`analysis_mode`, `camera_ok`	Mode switch confirmation
REST Endpoints
Method	Endpoint	Body	Description
`POST`	`/api/start`	`{simulation, analysis_mode}`	Start session
`POST`	`/api/stop`	—	Stop session
`POST`	`/api/pause`	—	Pause Gemma inference
`POST`	`/api/resume`	—	Resume inference
`POST`	`/api/set-mode`	`{analysis_mode}`	Switch mode live
`POST`	`/api/analyze-image`	`{image_b64}`	One-shot visual analysis
`GET`	`/api/snapshot`	—	Latest camera frame as base64
`GET`	`/api/report?hours=N`	—	Download PDF report
`GET`	`/api/status`	—	System health check
`GET`	`/api/history`	—	Last 50 alerts
---
Gemma 4 Integration In Depth
1. Structured Audio Prompt
Audio features extracted by numpy DSP are described in natural language and sent to Gemma 4 as a clinical sensor report:
```
=== EchoGlow Sensor Report — 14:22:07 ===
Cry detected     : YES — ongoing for 8.3 s
Energy (RMS)     : 0.7200  [HIGH]
Pitch estimate   : 510.0 Hz
Spectral centroid: 520.0 Hz
Zero-crossing    : 0.1800
Intensity pattern: escalating

Audio observation:
High-pitched escalating cry at ~510 Hz. High intensity. Pain pattern.
Sudden onset with rapid escalation consistent with acute distress.
```
2. Vision Input (Multimodal)
When a camera frame is available, it is base64-encoded and included in the Ollama API call alongside the audio description:
```python
payload = {
    "model": "gemma4:4b",
    "messages": [
        {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": audio_prompt,
            "images": [base64_frame]   # Gemma 4 native vision
        }
    ],
    "format": "json",   # Ollama-level JSON constraint
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 512}
}
```
3. Structured Output
Ollama's `format: json` parameter combined with a strict system prompt and a three-pass JSON parser guarantees structured output even if the model adds conversational text:
```json
{
  "cry_classification": "pain",
  "severity": "urgent",
  "confidence": 0.87,
  "reasoning": "High-pitched escalating cry at 510 Hz with sudden onset. Acoustic characteristics are highly indicative of acute pain in a neonate.",
  "clinical_note": "Acute pain-type cry detected. Source of pain unknown.",
  "action_recommended": "Perform full pain assessment (APGAR, pain scale). Check IV site, umbilical stump, positioning.",
  "respiratory_concern": false
}
```
The three-pass parser attempts: (1) direct `json.loads`, (2) first `{...}` block extraction, (3) largest `{...}` block extraction — before falling back to a rule-based classifier.
4. Rule-Based Fallback
If Ollama is unavailable or returns unparseable output, `gemma_analyzer.py` falls back to a deterministic classifier based on the audio features (RMS, pitch, pattern). The UI clearly labels results as `◆ Fallback` vs `✦ Gemma 4`.
---
Hardware Bill of Materials
Full clinic deployment — no internet required after setup:
Component	Recommended	Cost
SBC	Raspberry Pi 5 (8 GB)	~$80
Microphone	USB cardioid (e.g. Blue Snowball)	~$15
Webcam	USB 1080p (e.g. Logitech C270)	~$25
Storage	32 GB microSD (Class 10)	~$8
Power	Official Pi 5 USB-C PSU	~$12
Case	Raspberry Pi 5 case with fan	~$10
Total		~$150
Compared to: $8,000–$15,000 for commercial NICU multi-parameter monitors.
Development / testing on any machine with:
4+ GB VRAM (for E4B) or 2+ GB (for E2B)
Python 3.10+
Ollama installed
---
Troubleshooting
`RuntimeError: Directory 'static' does not exist`
The app uses absolute paths. Make sure you run from the project root:
```bash
cd EchoGlow
python app.py   # not: python path/to/EchoGlow/app.py
```
`404 Not Found on /api/chat`
Two possible causes:
Wrong model name — run `ollama list` and update `GEMMA_MODEL` in `config.py` to match exactly.
Old Ollama version — the code automatically falls back to `/api/generate`. Update Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
`405 Method Not Allowed`
The POST endpoints require a JSON body. If testing manually:
```bash
curl -X POST http://localhost:8765/api/start \
  -H "Content-Type: application/json" \
  -d '{"simulation": true}'
```
`WARNING: No JSON block found in Gemma response`
Gemma 4 wrapped its answer in conversational text. The three-pass parser handles most cases. If it persists:
Check `GEMMA_MODEL` matches your downloaded model exactly
Ensure `"format": "json"` is supported by your Ollama version (`ollama --version` ≥ 0.1.14)
Microphone not detected
EchoGlow automatically switches to Demo Mode (simulation) when no microphone is found. You can also force it:
```bash
# Dashboard toggle: DEMO ↔ LIVE
# or via API:
curl -X POST http://localhost:8765/api/start \
  -H "Content-Type: application/json" \
  -d '{"simulation": true}'
```
PDF report empty / `fpdf2` error
```bash
pip install fpdf2
```
---
Roadmap
[ ] Raspberry Pi OS installer — one-command setup script
[ ] Bluetooth SpO₂ integration — pair with pulse-oximeter for combined vitals
[ ] Multi-crib mode — monitor up to 4 audio sources on a single device
[ ] Alert escalation — SMS/WhatsApp notification via local gateway
[ ] Clinical validation study — pilot with community health programme in Telangana, India
[ ] Android APK — native packaging using Gemma 4 LiteRT for on-device inference
[ ] Fine-tuned audio features — train on validated infant cry datasets (Donate-a-Cry corpus)
[ ] FHIR integration — export alerts directly to electronic health records
---
Disclaimer
EchoGlow is a research prototype and AI-assisted monitoring aid. It does not replace qualified clinical judgment. All alerts must be evaluated by competent medical personnel. EchoGlow is not approved for clinical use and should not be used as the sole basis for any medical decision.
Patient audio and images are processed entirely on-device and are never transmitted to any external server.
---
Contributing
Contributions are welcome. Please open an issue before submitting a pull request for significant changes.
```bash
# Fork the repo, then:
git checkout -b feature/my-feature
git commit -m "feat: describe your change"
git push origin feature/my-feature
# Open a pull request
```
---
License
MIT License — see LICENSE for details.
---
Acknowledgements
Google DeepMind — Gemma 4 open weights
Ollama — local LLM serving infrastructure
FastAPI — async Python web framework
fpdf2 — PDF generation
OpenCV — computer vision
Built for the Gemma 4 Good Hackathon — Health & Sciences Track
---
<div align="center">
EchoGlow · Built with ❤️ for the clinics that need it most
github.com/quartzap/EchoGlow
</div>
