"""
gemma_analyzer.py
─────────────────
Sends audio features (and an optional camera frame) to Gemma 4 via the
local Ollama API and returns a structured clinical analysis.
"""

import json
import logging
import re
import time
from typing import Optional

import requests

from audio_processor import AudioFeatures
from config import GEMMA_MODEL, OLLAMA_URL

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are EchoGlow, an AI clinical-support module embedded in a neonatal ICU monitoring device.
Your sole function is to analyse infant audio/visual data and return structured clinical guidance to nursing staff.

STRICT OUTPUT RULE: respond ONLY with a single valid JSON object — no preamble, no markdown fences, no extra text.

JSON schema:
{
  "cry_classification": "<one of: calm | hunger | discomfort | pain | stridor | wheezing | respiratory_distress | unknown>",
  "severity":           "<one of: normal | monitor | urgent | critical>",
  "confidence":         <float 0.0–1.0>,
  "reasoning":          "<2–3 concise clinical sentences>",
  "clinical_note":      "<text suitable for nursing notes, ≤40 words>",
  "action_recommended": "<specific, actionable instruction for care staff>",
  "respiratory_concern": <true | false>
}

Classification reference:
  calm               → no vocalization, settled infant
  hunger             → rhythmic, regular cry; 300–420 Hz; periodic pauses
  discomfort         → fussy, irregular, low-moderate intensity
  pain               → high-pitched (>450 Hz), sudden onset, escalating, minimal pauses
  stridor            → harsh inspiratory noise; possible partial airway obstruction — URGENT
  wheezing           → expiratory musical sound; possible lower-airway involvement
  respiratory_distress → any abnormal breathing pattern requiring immediate evaluation
  unknown            → insufficient signal for classification

Severity mapping:
  normal   → calm / minor discomfort, no intervention required
  monitor  → hunger / mild discomfort, log and check within 10 min
  urgent   → pain / unexplained distress, assess within 2 min
  critical → stridor / wheezing / respiratory_distress, immediate response required"""


def _build_user_prompt(feat: AudioFeatures, has_image: bool) -> str:
    cry_line = (
        f"YES — ongoing for {feat.cry_duration:.1f} s"
        if feat.cry_detected
        else "NO"
    )
    energy_desc = (
        "HIGH" if feat.rms_energy > 0.55
        else "MODERATE" if feat.rms_energy > 0.2
        else "LOW"
    )
    lines = [
        f"=== EchoGlow Sensor Report — {time.strftime('%H:%M:%S')} ===",
        f"Cry detected     : {cry_line}",
        f"Energy (RMS)     : {feat.rms_energy:.4f}  [{energy_desc}]",
        f"Pitch estimate   : {feat.pitch_estimate:.1f} Hz",
        f"Spectral centroid: {feat.spectral_centroid:.1f} Hz",
        f"Zero-crossing    : {feat.zero_crossing_rate:.4f}",
        f"Silence ratio    : {feat.silence_ratio:.3f}",
        f"Intensity pattern: {feat.intensity_pattern}",
        "",
        "Audio observation:",
        feat.description,
    ]
    if has_image:
        lines += [
            "",
            "A camera frame from above the crib is attached.",
            "Note any visible signs: skin colour (pallor/cyanosis), "
            "chest movement, body posture, or agitation.",
        ]
    lines += ["", "Analyse the above and return your JSON response."]
    return "\n".join(lines)


# ── Analyzer class ────────────────────────────────────────────────────────────

class GemmaAnalyzer:

    _TIMEOUT = 30  # seconds

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if r.status_code != 200:
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            # Match on base name (ignore tag) so gemma4:latest also matches
            base = GEMMA_MODEL.split(":")[0].lower()
            return any(base in m.lower() for m in models)
        except Exception:
            return False

    def _get_running_model(self) -> str:
        """Return the best matching model name from what Ollama actually has."""
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if r.status_code != 200:
                return GEMMA_MODEL
            models = [m["name"] for m in r.json().get("models", [])]
            base = GEMMA_MODEL.split(":")[0].lower()
            matches = [m for m in models if base in m.lower()]
            return matches[0] if matches else GEMMA_MODEL
        except Exception:
            return GEMMA_MODEL

    def analyze(
        self,
        feat: AudioFeatures,
        image_b64: Optional[str] = None,
    ) -> dict:
        """Call Gemma 4 and return a parsed analysis dict."""
        try:
            return self._call_ollama(feat, image_b64)
        except Exception as exc:
            logger.warning(f"Gemma call failed ({exc}), using rule-based fallback")
            return self._rule_based_fallback(feat)

    # ── Ollama call ───────────────────────────────────────────────────────────

    def _call_ollama(self, feat: AudioFeatures, image_b64: Optional[str]) -> dict:
        user_prompt = _build_user_prompt(feat, has_image=bool(image_b64))
        model = self._get_running_model()

        # Try /api/chat first (Ollama ≥ 0.1.14), fall back to /api/generate
        try:
            raw = self._call_chat(model, user_prompt, image_b64)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.warning("/api/chat not found — falling back to /api/generate")
                raw = self._call_generate(model, user_prompt, image_b64)
            else:
                raise

        return self._parse_response(raw, feat)

    def _call_chat(self, model: str, user_prompt: str, image_b64: Optional[str]) -> str:
        """Returns raw response string. Raises HTTPError on failure."""
        message: dict = {"role": "user", "content": user_prompt}
        if image_b64:
            message["images"] = [image_b64]

        payload = {
            "model":    model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                message,
            ],
            "stream":  False,
            "format":  "json",          # forces Ollama to emit valid JSON
            "options": {"temperature": 0.1, "num_predict": 512},
        }

        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=self._TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        logger.debug(f"Gemma raw response: {raw[:300]}")
        return raw

    def _call_generate(self, model: str, user_prompt: str, image_b64: Optional[str]) -> str:
        """Fallback for older Ollama versions that use /api/generate."""
        full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"
        payload = {
            "model":  model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",           # forces Ollama to emit valid JSON
            "options": {"temperature": 0.1, "num_predict": 512},
        }
        if image_b64:
            payload["images"] = [image_b64]

        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=self._TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        logger.debug(f"Gemma raw response: {raw[:300]}")
        return raw

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, raw: str, feat: AudioFeatures) -> dict:
        # Strip markdown fences and whitespace
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Try direct JSON parse first (works when format=json is respected)
        try:
            parsed = json.loads(cleaned)
            return self._sanitise(parsed, feat)
        except json.JSONDecodeError:
            pass

        # Try to find the first {...} block anywhere in the text
        m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                return self._sanitise(parsed, feat)
            except json.JSONDecodeError:
                pass

        # Try the largest {...} block (handles nested braces)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                return self._sanitise(parsed, feat)
            except json.JSONDecodeError:
                pass

        # Nothing worked — log full response so user can see what Gemma said
        logger.warning(
            f"No JSON block found in Gemma response. "
            f"Full response was:\n{raw[:600]}"
        )
        return self._rule_based_fallback(feat)

    def _sanitise(self, parsed: dict, feat: AudioFeatures) -> dict:
        """Validate and clean up a parsed dict from Gemma."""

        # Validate/sanitise fields
        allowed_classes = {
            "calm", "hunger", "discomfort", "pain",
            "stridor", "wheezing", "respiratory_distress", "unknown",
        }
        allowed_severity = {"normal", "monitor", "urgent", "critical"}

        parsed["cry_classification"] = parsed.get(
            "cry_classification", "unknown"
        ).lower()
        if parsed["cry_classification"] not in allowed_classes:
            parsed["cry_classification"] = "unknown"

        parsed["severity"] = parsed.get("severity", "normal").lower()
        if parsed["severity"] not in allowed_severity:
            parsed["severity"] = "normal"

        parsed["confidence"]         = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
        parsed["reasoning"]          = str(parsed.get("reasoning", ""))
        parsed["clinical_note"]      = str(parsed.get("clinical_note", ""))
        parsed["action_recommended"] = str(parsed.get("action_recommended", "Continue monitoring."))
        parsed["respiratory_concern"]= bool(parsed.get("respiratory_concern", False))
        parsed["source"]             = "gemma"
        return parsed

    # ── Rule-based fallback ───────────────────────────────────────────────────

    @staticmethod
    def _rule_based_fallback(feat: AudioFeatures) -> dict:
        """
        Simple heuristic used when Ollama is unavailable.
        Good enough for a demo and keeps the UI functional.
        """
        rms   = feat.rms_energy
        pitch = feat.pitch_estimate
        pat   = feat.intensity_pattern
        desc  = feat.description.lower()

        # Classify
        if not feat.cry_detected or rms < 0.02:
            cls, sev, conf = "calm", "normal", 0.92
            reason = "No infant vocalization detected. Infant appears settled."
            note   = "No distress signal. Continue routine monitoring."
            action = "No action required. Check again at next scheduled interval."
            resp   = False

        elif any(k in desc for k in ("stridor", "airway", "obstruction")):
            cls, sev, conf = "stridor", "critical", 0.87
            reason = (
                "Harsh inspiratory noise detected consistent with stridor. "
                "Possible partial upper-airway obstruction. Immediate clinical evaluation required."
            )
            note   = "Stridor-like sound detected. Urgent airway assessment needed."
            action = "IMMEDIATE: Notify senior nurse/physician. Assess airway patency now."
            resp   = True

        elif any(k in desc for k in ("wheez", "expiratory", "lower airway")):
            cls, sev, conf = "wheezing", "critical", 0.82
            reason = (
                "Expiratory wheeze pattern detected suggesting possible lower-airway involvement. "
                "Bronchospasm or secretion obstruction possible."
            )
            note   = "Expiratory wheeze detected. Lower-airway assessment required."
            action = "URGENT: Alert medical team. Assess respiratory status immediately."
            resp   = True

        elif pitch > 460 or pat == "escalating" or rms > 0.65:
            cls, sev, conf = "pain", "urgent", 0.80
            reason = (
                f"High-pitched escalating cry at {pitch:.0f} Hz with {pat} pattern. "
                "Acoustic signature consistent with acute pain or significant discomfort."
            )
            note   = f"High-intensity pain-like cry ({pitch:.0f} Hz). Assess for cause."
            action = "Assess infant within 2 minutes. Check IV site, position, recent procedures."
            resp   = False

        elif pat == "rhythmic" and 280 <= pitch <= 430:
            cls, sev, conf = "hunger", "monitor", 0.78
            reason = (
                f"Regular rhythmic cry at {pitch:.0f} Hz with periodic pause-pattern. "
                "Acoustic characteristics typical of hunger vocalization."
            )
            note   = "Hunger-pattern cry. Check feeding schedule."
            action = "Check last feed time. Offer feed or notify parent/nurse."
            resp   = False

        else:
            cls, sev, conf = "discomfort", "monitor", 0.65
            reason = (
                "Irregular fussy vocalization detected. Non-specific discomfort pattern. "
                "Cause unclear from audio alone."
            )
            note   = "General discomfort cry. Assess for common causes."
            action = "Check nappy, position, temperature. Comfort intervention suggested."
            resp   = False

        return {
            "cry_classification":  cls,
            "severity":            sev,
            "confidence":          conf,
            "reasoning":           reason,
            "clinical_note":       note,
            "action_recommended":  action,
            "respiratory_concern": resp,
            "source":              "fallback",
        }
