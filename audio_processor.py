"""
audio_processor.py
──────────────────
Handles microphone capture (via sounddevice) and audio feature extraction.
Falls back to a clinically-realistic simulation when no microphone is found.
"""

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import (
    CRY_PITCH_MAX_HZ,
    CRY_PITCH_MIN_HZ,
    CRY_RMS_THRESHOLD,
    HEARTBEAT_INTERVAL,
    SAMPLE_RATE,
    WINDOW_SIZE,
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class AudioFeatures:
    timestamp:         float
    rms_energy:        float   # 0-1 normalised loudness
    zero_crossing_rate: float  # voiced / unvoiced indicator
    spectral_centroid: float   # Hz – pitch-related brightness
    spectral_rolloff:  float   # Hz – upper-energy boundary
    pitch_estimate:    float   # Hz – dominant fundamental
    cry_detected:      bool
    cry_duration:      float   # seconds
    silence_ratio:     float   # fraction of silence in window
    intensity_pattern: str     # quiet/rhythmic/continuous/escalating/prolonged
    description:       str     # plain-English summary for Gemma


# ── Simulation scenarios ──────────────────────────────────────────────────────

_SIMULATION_SCRIPT = [
    # (duration_s, label, rms, pitch, pattern, desc)
    (12, "calm",
     0.01, 0, "quiet",
     "No vocalization. Infant appears settled. Ambient room sounds only."),

    (22, "hunger",
     0.48, 380, "rhythmic",
     "Rhythmic, medium-pitched cry at ~380 Hz with regular 2–3 s cry/pause "
     "cadence. Moderate intensity. Pattern consistent with hunger."),

    (8, "calm",
     0.01, 0, "quiet",
     "Infant briefly self-soothed. Room quiet."),

    (18, "pain",
     0.82, 530, "escalating",
     "High-pitched escalating cry at ~530 Hz. Sudden onset, high intensity. "
     "Minimal pauses. Sharp, urgent vocalization consistent with acute pain."),

    (6, "discomfort",
     0.35, 310, "intermittent",
     "Fussy, irregular cry at ~310 Hz. Intermittent with self-soothing attempts. "
     "Low-moderate intensity suggesting general discomfort."),

    (20, "stridor",
     0.68, 285, "continuous",
     "ALERT: Harsh, high-pitched inspiratory stridor detected at ~285 Hz. "
     "Continuous noisy breathing with reduced cry pauses. Possible partial "
     "upper-airway obstruction. Requires immediate assessment."),

    (12, "calm",
     0.01, 0, "quiet",
     "Infant calm following intervention. No vocalization detected."),

    (15, "wheezing",
     0.52, 240, "continuous",
     "Low-pitched expiratory wheeze at ~240 Hz. Musical, continuous quality. "
     "Possible lower airway involvement. Breath sounds abnormal."),

    (10, "calm",
     0.01, 0, "quiet",
     "Room quiet. No vocalization or respiratory distress sounds."),
]


# ── Main class ────────────────────────────────────────────────────────────────

class AudioProcessor:

    def __init__(self):
        self.sample_rate  = SAMPLE_RATE
        self.buf_samples  = int(SAMPLE_RATE * WINDOW_SIZE)
        self._buffer      = np.zeros(self.buf_samples, dtype=np.float32)
        self._lock        = threading.Lock()

        self.is_running       = False
        self.has_mic          = False
        self.simulation_mode  = False

        self._cry_start: Optional[float] = None
        self._feature_queue: queue.Queue = queue.Queue(maxsize=100)
        self._stream = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, simulation: bool = False):
        self.simulation_mode = simulation
        self.is_running = True

        if not simulation:
            self._try_start_mic()

        if self.simulation_mode:
            self._start_simulation_thread()

    def stop(self):
        self.is_running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def extract_features(self) -> Optional[AudioFeatures]:
        """Return the latest feature set (non-blocking)."""
        if self.simulation_mode:
            try:
                return self._feature_queue.get_nowait()
            except queue.Empty:
                return None

        with self._lock:
            buf = self._buffer.copy()

        return self._compute_features(buf)

    # ── Microphone ────────────────────────────────────────────────────────────

    def _try_start_mic(self):
        try:
            import sounddevice as sd  # noqa: F401 – lazy import

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(self.sample_rate * 0.1),
                callback=self._audio_callback,
            )
            self._stream.start()
            self.has_mic = True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"Microphone unavailable ({exc}). Switching to simulation mode."
            )
            self.simulation_mode = True

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0]
        with self._lock:
            self._buffer = np.roll(self._buffer, -len(chunk))
            self._buffer[-len(chunk):] = chunk

    # ── Simulation ────────────────────────────────────────────────────────────

    def _start_simulation_thread(self):
        t = threading.Thread(target=self._simulation_loop, daemon=True)
        t.start()

    def _simulation_loop(self):
        rng = np.random.default_rng(42)
        script = _SIMULATION_SCRIPT
        idx = 0

        while self.is_running:
            dur, label, base_rms, base_pitch, pattern, desc = script[idx % len(script)]
            idx += 1
            deadline = time.time() + dur

            while time.time() < deadline and self.is_running:
                # Add subtle jitter so waveform looks alive
                rms   = max(0.0, base_rms   + rng.normal(0, base_rms * 0.12))
                pitch = max(0.0, base_pitch + rng.normal(0, max(base_pitch * 0.05, 1)))

                # Cry start/stop tracking
                if rms > CRY_RMS_THRESHOLD:
                    if self._cry_start is None:
                        self._cry_start = time.time()
                else:
                    self._cry_start = None

                cry_dur = (time.time() - self._cry_start) if self._cry_start else 0.0

                zcr      = 0.16 + rng.normal(0, 0.03) if pitch > 0 else 0.05
                centroid = pitch * 2.3 + rng.normal(0, 20) if pitch > 0 else 200.0
                rolloff  = pitch * 3.8 + rng.normal(0, 30) if pitch > 0 else 350.0
                silence  = max(0.0, 0.9 - rms * 1.8)

                feat = AudioFeatures(
                    timestamp         = time.time(),
                    rms_energy        = round(float(rms),     4),
                    zero_crossing_rate= round(float(zcr),     4),
                    spectral_centroid = round(float(centroid), 1),
                    spectral_rolloff  = round(float(rolloff),  1),
                    pitch_estimate    = round(float(pitch),    1),
                    cry_detected      = rms > CRY_RMS_THRESHOLD,
                    cry_duration      = round(cry_dur, 1),
                    silence_ratio     = round(float(silence), 3),
                    intensity_pattern = pattern,
                    description       = desc,
                )

                try:
                    self._feature_queue.put_nowait(feat)
                except queue.Full:
                    pass

                time.sleep(HEARTBEAT_INTERVAL)

    # ── DSP (real microphone path) ────────────────────────────────────────────

    def _compute_features(self, buf: np.ndarray) -> Optional[AudioFeatures]:
        if buf.size == 0:
            return None

        # RMS
        rms = float(math.sqrt(np.mean(buf ** 2)))

        # Zero-crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(buf)))) / 2)

        # FFT
        fft  = np.abs(np.fft.rfft(buf * np.hanning(len(buf))))
        freqs = np.fft.rfftfreq(len(buf), 1.0 / self.sample_rate)

        total_e = np.sum(fft)
        if total_e > 1e-9:
            centroid = float(np.dot(freqs, fft) / total_e)
            cumsum   = np.cumsum(fft)
            ri       = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            rolloff  = float(freqs[min(ri, len(freqs) - 1)])
        else:
            centroid = rolloff = 0.0

        # Dominant pitch in infant-cry band
        mask = (freqs >= CRY_PITCH_MIN_HZ) & (freqs <= CRY_PITCH_MAX_HZ)
        if mask.any() and fft[mask].sum() > 1e-9:
            pitch = float(freqs[mask][np.argmax(fft[mask])])
        else:
            pitch = 0.0

        cry_detected = rms > CRY_RMS_THRESHOLD and pitch > CRY_PITCH_MIN_HZ

        if cry_detected:
            if self._cry_start is None:
                self._cry_start = time.time()
        else:
            self._cry_start = None

        cry_dur = (time.time() - self._cry_start) if self._cry_start else 0.0

        # Silence ratio (frame-level)
        frame = 160
        frames_rms = [
            math.sqrt(max(0, float(np.mean(buf[i: i + frame] ** 2))))
            for i in range(0, len(buf) - frame, frame)
        ]
        silence = float(np.mean([r < 0.008 for r in frames_rms])) if frames_rms else 1.0

        # Intensity pattern
        if not cry_detected:
            pattern = "quiet"
        elif cry_dur > 45:
            pattern = "prolonged"
        elif zcr > 0.22:
            pattern = "rhythmic"
        elif rms > 0.6:
            pattern = "escalating"
        else:
            pattern = "continuous"

        desc = self._build_description(rms, pitch, pattern, zcr, centroid, cry_dur)

        return AudioFeatures(
            timestamp         = time.time(),
            rms_energy        = round(rms,      4),
            zero_crossing_rate= round(zcr,      4),
            spectral_centroid = round(centroid, 1),
            spectral_rolloff  = round(rolloff,  1),
            pitch_estimate    = round(pitch,    1),
            cry_detected      = cry_detected,
            cry_duration      = round(cry_dur,  1),
            silence_ratio     = round(silence,  3),
            intensity_pattern = pattern,
            description       = desc,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_description(rms, pitch, pattern, zcr, centroid, cry_dur) -> str:
        if rms < CRY_RMS_THRESHOLD:
            level = "low" if rms < 0.01 else "moderate"
            return f"No infant vocalization. Room ambient noise level: {level}."

        intensity = "high-intensity" if rms > 0.55 else "moderate-intensity" if rms > 0.25 else "low-intensity"
        pitch_d   = "high-pitched" if pitch > 440 else "mid-pitched" if pitch > 280 else "low-pitched"
        out = f"{intensity} {pitch_d} cry detected at ~{int(pitch)} Hz, {pattern} pattern."

        if cry_dur > 8:
            out += f" Continuous vocalization for {int(cry_dur)} s."
        if centroid > 500:
            out += " High spectral energy – possible pain or acute distress."
        if zcr > 0.25:
            out += " Irregular ZCR suggests laboured or abnormal breathing pattern."
        return out
