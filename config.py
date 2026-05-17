import os

# ── Ollama / Gemma ──────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
GEMMA_MODEL  = os.getenv("GEMMA_MODEL",  "gemma4:e4b")   # change to gemma4:2b for E2B

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000   # Hz
WINDOW_SIZE        = 3.0     # seconds kept in circular buffer
ANALYSIS_INTERVAL  = 7.0     # seconds between Gemma inference calls
HEARTBEAT_INTERVAL = 0.08    # seconds between waveform pushes (~12 fps)

# Cry-detection thresholds
CRY_RMS_THRESHOLD   = 0.035
CRY_PITCH_MIN_HZ    = 150
CRY_PITCH_MAX_HZ    = 900

# ── Server ───────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8766

APP_VERSION = "1.0.0"
