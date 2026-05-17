"""
app.py — EchoGlow Neonatal Monitor
───────────────────────────────────
FastAPI server:
  GET  /                → dashboard HTML
  WS   /ws              → real-time analysis stream
  POST /api/start       → begin monitoring session
  POST /api/stop        → end session
  POST /api/pause       → pause Gemma analysis (waveform keeps running)
  POST /api/resume      → resume Gemma analysis
  POST /api/analyze-image → one-shot image analysis via Gemma vision
  GET  /api/status      → system health
  GET  /api/history     → alert history (last 50)
  GET  /api/report      → download PDF report (?hours=1|7|12|24)
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional, Set

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from audio_processor import AudioProcessor
from config import ANALYSIS_INTERVAL, GEMMA_MODEL, HEARTBEAT_INTERVAL, HOST, PORT
from gemma_analyzer import GemmaAnalyzer
from vision_processor import VisionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("echoglow")

# Show Gemma raw responses in terminal to help debug JSON issues
logging.getLogger("gemma_analyzer").setLevel(logging.DEBUG)

# ── Resolve paths relative to this file (works regardless of cwd) ─────────────
BASE_DIR    = Path(__file__).resolve().parent
STATIC_DIR  = BASE_DIR / "static"
INDEX_HTML  = STATIC_DIR / "index.html"

# ── App & singletons ──────────────────────────────────────────────────────────

app = FastAPI(title="EchoGlow Neonatal Monitor", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

audio_proc  = AudioProcessor()
gemma       = GemmaAnalyzer()
vision_proc = VisionProcessor()

# ── Session state ─────────────────────────────────────────────────────────────

session: dict = {
    "is_active":      False,
    "start_time":     None,
    "mode":           "idle",        # idle | live | simulation
    "paused":         False,
    "analysis_mode":  "audio",       # audio | video | both
    "alerts":         [],
    "last_analysis":  None,
    "analysis_count": 0,
    "use_camera":     False,
    "last_frame":     None,          # most-recent camera b64 (set during analysis)
}

active_ws: Set[WebSocket] = set()
_monitor_task: Optional[asyncio.Task] = None

# Waveform ring buffer (300 points → ~25 s at 12 fps)
_waveform: list = [0.0] * 300


# ── WebSocket helpers ─────────────────────────────────────────────────────────

async def broadcast(data: dict):
    dead: Set[WebSocket] = set()
    for ws in list(active_ws):
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    active_ws.difference_update(dead)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(INDEX_HTML))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_ws.add(websocket)
    logger.info(f"WS client connected ({len(active_ws)} total)")

    # Send initial state snapshot
    await websocket.send_json({
        "type":       "state",
        "session":    _session_snapshot(),
        "waveform":   _waveform[-100:],
        "ollama_ok":  gemma.is_available(),
        "mic_ok":     audio_proc.has_mic,
        "camera_ok":  vision_proc.is_available(),
        "model":      GEMMA_MODEL,
        "paused":     session["paused"],
    })

    try:
        while True:
            await asyncio.sleep(60)   # keep-alive; messages driven server→client
    except WebSocketDisconnect:
        pass
    finally:
        active_ws.discard(websocket)
        logger.info(f"WS client disconnected ({len(active_ws)} remaining)")


@app.post("/api/start")
async def start_session(request: Request):
    global _monitor_task
    try:
        body = await request.json()
    except Exception:
        body = {}

    if session["is_active"]:
        return JSONResponse({"status": "already_running", "mode": session["mode"]})

    want_sim    = body.get("simulation", False)
    use_camera  = body.get("camera", False)

    # Force simulation when no mic
    audio_proc.start(simulation=want_sim)
    actual_sim = want_sim or audio_proc.simulation_mode

    session.update({
        "is_active":      True,
        "start_time":     time.time(),
        "mode":           "simulation" if actual_sim else "live",
        "paused":         False,
        "analysis_mode":  body.get("analysis_mode", "audio"),
        "alerts":         [],
        "last_analysis":  None,
        "analysis_count": 0,
        "use_camera":     vision_proc.is_available(),
        "last_frame":     None,
    })

    _monitor_task = asyncio.create_task(_monitoring_loop())
    logger.info(f"Session started — mode={session['mode']}, camera={session['use_camera']}")
    return {"status": "started", "mode": session["mode"]}


@app.post("/api/stop")
async def stop_session():
    if not session["is_active"]:
        return {"status": "not_running"}

    session["is_active"] = False
    audio_proc.stop()

    if _monitor_task and not _monitor_task.done():
        _monitor_task.cancel()

    logger.info(f"Session stopped. Alerts recorded: {len(session['alerts'])}")
    await broadcast({"type": "stopped", "alert_count": len(session["alerts"])})
    return {"status": "stopped", "alerts": len(session["alerts"])}


@app.get("/api/status")
async def get_status():
    return {
        "session":     session["is_active"],
        "paused":      session["paused"],
        "mode":        session["mode"],
        "ollama":      gemma.is_available(),
        "mic":         audio_proc.has_mic,
        "camera":      vision_proc.is_available(),
        "model":       GEMMA_MODEL,
        "alert_count": len(session["alerts"]),
        "duration":    int(time.time() - session["start_time"]) if session["start_time"] else 0,
    }


@app.get("/api/history")
async def get_history():
    return {"alerts": session["alerts"][-50:]}


@app.post("/api/pause")
async def pause_session():
    """Suspend Gemma analysis — waveform keeps streaming."""
    if not session["is_active"]:
        return JSONResponse({"status": "not_running"}, status_code=400)
    if session["paused"]:
        return {"status": "already_paused"}
    session["paused"] = True
    logger.info("Session paused")
    await broadcast({"type": "paused", "timestamp": time.time()})
    return {"status": "paused"}


@app.post("/api/resume")
async def resume_session():
    """Resume Gemma analysis after a pause."""
    if not session["is_active"]:
        return JSONResponse({"status": "not_running"}, status_code=400)
    if not session["paused"]:
        return {"status": "not_paused"}
    session["paused"] = False
    logger.info("Session resumed")
    await broadcast({"type": "resumed", "timestamp": time.time()})
    return {"status": "resumed"}


@app.post("/api/set-mode")
async def set_analysis_mode(request: Request):
    """Switch analysis mode (audio / video / both) mid-session."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    mode = body.get("analysis_mode", "audio")
    if mode not in ("audio", "video", "both"):
        return JSONResponse({"error": "Invalid mode. Use audio|video|both"}, status_code=400)
    session["analysis_mode"] = mode
    logger.info(f"Analysis mode switched to: {mode}")
    await broadcast({
        "type":          "mode_changed",
        "analysis_mode": mode,
        "camera_ok":     vision_proc.is_available(),
    })
    return {"status": "ok", "analysis_mode": mode}


@app.get("/api/snapshot")
async def snapshot():
    """Return the latest camera frame as base64, or the last captured frame."""
    # Try live capture first
    if vision_proc.is_available():
        frame = vision_proc.capture_frame()
        if frame:
            session["last_frame"] = frame
            return {"image_b64": frame, "source": "live"}

    # Fall back to last stored frame
    if session.get("last_frame"):
        return {"image_b64": session["last_frame"], "source": "cached"}

    return JSONResponse({"error": "No camera and no cached frame available"}, status_code=404)


@app.post("/api/analyze-image")
async def analyze_image(request: Request):
    """Analyze an uploaded photo via Gemma 4 vision (no audio required)."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    image_b64 = data.get("image_b64", "")
    if not image_b64:
        return JSONResponse({"error": "No image provided"}, status_code=400)

    from audio_processor import AudioFeatures
    feat = AudioFeatures(
        timestamp         = time.time(),
        rms_energy        = 0.0,
        zero_crossing_rate= 0.0,
        spectral_centroid = 0.0,
        spectral_rolloff  = 0.0,
        pitch_estimate    = 0.0,
        cry_detected      = False,
        cry_duration      = 0.0,
        silence_ratio     = 1.0,
        intensity_pattern = "quiet",
        description       = (
            "No audio data available. This is a visual-only analysis. "
            "Please assess the uploaded image for signs of infant distress, "
            "skin colour changes, posture, chest movement, or agitation."
        ),
    )
    loop = asyncio.get_event_loop()
    analysis = await loop.run_in_executor(None, gemma.analyze, feat, image_b64)

    # Always store image analyses in the alert log regardless of severity
    now = time.time()
    session["alerts"].append({
        **analysis,
        "timestamp":   now,
        "source_type": "image",   # distinguishes from audio alerts in the feed
        "cry_duration": 0.0,
        "pitch":        0.0,
        "rms":          0.0,
    })

    # Broadcast to all WS clients so the main panel updates immediately
    await broadcast({
        "type":      "image_analysis",
        "timestamp": now,
        "analysis":  analysis,
        "alert_count": len(session["alerts"]),
    })

    return analysis


@app.get("/api/report")
async def download_report(hours: int = 1):
    """Generate and stream a PDF clinical summary report."""
    from fastapi.responses import Response
    from report_generator import ReportGenerator

    now    = time.time()
    cutoff = now - hours * 3600
    filtered = [a for a in session["alerts"] if a.get("timestamp", 0) >= cutoff]

    rg = ReportGenerator()
    try:
        pdf_bytes = rg.generate(
            alerts       = filtered,
            hours        = hours,
            session      = {**session, "model": GEMMA_MODEL},
            generated_at = now,
        )
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    fname = f"echoglow_report_{hours}h_{int(now)}.pdf"
    return Response(
        content = pdf_bytes,
        media_type = "application/pdf",
        headers = {"Content-Disposition": f"attachment; filename={fname}"},
    )


# ── Monitoring loop ───────────────────────────────────────────────────────────

async def _monitoring_loop():
    import dataclasses
    global _waveform
    last_gemma_time = 0.0
    loop = asyncio.get_event_loop()

    while session["is_active"]:
        try:
            # Always capture audio (needed for waveform regardless of mode)
            feat = await loop.run_in_executor(None, audio_proc.extract_features)
            if feat is None:
                await asyncio.sleep(0.05)
                continue

            _waveform.append(round(feat.rms_energy, 4))
            if len(_waveform) > 300:
                _waveform = _waveform[-300:]

            now     = time.time()
            elapsed = int(now - session["start_time"]) if session["start_time"] else 0
            amode   = session.get("analysis_mode", "audio")

            # ── PAUSED ──────────────────────────────────────────────────────
            if session["paused"]:
                await broadcast({
                    "type":             "heartbeat",
                    "timestamp":        now,
                    "session_duration": elapsed,
                    "rms":              feat.rms_energy,
                    "cry_detected":     feat.cry_detected,
                    "pitch":            feat.pitch_estimate,
                    "waveform_point":   feat.rms_energy,
                    "paused":           True,
                })
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                continue

            # ── Full Gemma analysis (on interval) ────────────────────────────
            if now - last_gemma_time >= ANALYSIS_INTERVAL:
                last_gemma_time = now

                # ── Decide what to send Gemma based on analysis_mode ────────
                image_b64 = None

                if amode in ("video", "both"):
                    if vision_proc.is_available():
                        image_b64 = await loop.run_in_executor(
                            None, vision_proc.capture_frame
                        )
                    if image_b64:
                        session["last_frame"] = image_b64

                # Adapt audio description for video-only mode
                if amode == "video":
                    feat = dataclasses.replace(
                        feat,
                        description=(
                            "Video-only analysis mode — no audio signal is used. "
                            "A camera frame is attached. Focus entirely on visual "
                            "signs of infant distress: skin colour, posture, chest "
                            "movement, agitation, or visible breathing difficulty."
                        ),
                        cry_detected=False,
                    )
                elif amode == "both" and image_b64:
                    feat = dataclasses.replace(
                        feat,
                        description=(
                            feat.description +
                            " A camera frame is also attached — combine audio and "
                            "visual cues for a more complete assessment."
                        ),
                    )

                analysis = await loop.run_in_executor(
                    None, gemma.analyze, feat, image_b64
                )

                session["last_analysis"]   = analysis
                session["analysis_count"] += 1

                if analysis.get("severity") not in ("normal",):
                    session["alerts"].append({
                        **analysis,
                        "timestamp":    now,
                        "cry_duration": feat.cry_duration,
                        "pitch":        feat.pitch_estimate,
                        "rms":          feat.rms_energy,
                    })

                await broadcast({
                    "type":             "analysis",
                    "timestamp":        now,
                    "session_duration": elapsed,
                    "alert_count":      len(session["alerts"]),
                    "analysis_mode":    amode,
                    "has_image":        bool(image_b64),
                    "frame":            image_b64 if image_b64 else None,
                    "features": {
                        "rms":          feat.rms_energy,
                        "pitch":        feat.pitch_estimate,
                        "pattern":      feat.intensity_pattern,
                        "cry_detected": feat.cry_detected,
                        "cry_duration": feat.cry_duration,
                        "zcr":          feat.zero_crossing_rate,
                        "centroid":     feat.spectral_centroid,
                    },
                    "analysis":  analysis,
                    "waveform":  _waveform[-120:],
                })

            else:
                # ── Heartbeat ────────────────────────────────────────────────
                await broadcast({
                    "type":             "heartbeat",
                    "timestamp":        now,
                    "session_duration": elapsed,
                    "rms":              feat.rms_energy,
                    "cry_detected":     feat.cry_detected,
                    "pitch":            feat.pitch_estimate,
                    "waveform_point":   feat.rms_energy,
                    "analysis_mode":    amode,
                })

            await asyncio.sleep(HEARTBEAT_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Monitoring loop error: {exc}", exc_info=True)
            await asyncio.sleep(1.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_snapshot() -> dict:
    return {
        "is_active":      session["is_active"],
        "mode":           session["mode"],
        "paused":         session["paused"],
        "analysis_mode":  session["analysis_mode"],
        "analysis_count": session["analysis_count"],
        "alert_count":    len(session["alerts"]),
        "duration":       int(time.time() - session["start_time"]) if session["start_time"] else 0,
        "last_analysis":  session["last_analysis"],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🩺  EchoGlow Neonatal Monitor  v1.0.0")
    print(f"    Model : {GEMMA_MODEL}  (via Ollama at {__import__('config').OLLAMA_URL})")
    print(f"    Open  : http://localhost:{PORT}\n")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False, log_level="warning")
