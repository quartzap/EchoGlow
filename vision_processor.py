"""
vision_processor.py
───────────────────
Optional camera frame capture for Gemma 4 multimodal analysis.
Gracefully disabled if OpenCV or a camera is unavailable.
"""

import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VisionProcessor:

    def __init__(self, camera_index: int = 0):
        self._cap        = None
        self._available  = False
        self._camera_idx = camera_index
        self._try_init()

    def _try_init(self):
        try:
            import cv2  # noqa: F401
            self._cv2 = cv2
            cap = cv2.VideoCapture(self._camera_idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap       = cap
                self._available = True
                logger.info(f"Camera {self._camera_idx} initialised.")
            else:
                cap.release()
                logger.info("No camera found. Vision input disabled.")
        except ImportError:
            logger.info("OpenCV not installed. Vision input disabled.")
        except Exception as exc:
            logger.info(f"Camera init failed ({exc}). Vision input disabled.")

    def is_available(self) -> bool:
        return self._available

    def capture_frame(self) -> Optional[str]:
        """
        Capture one frame and return it as a base64-encoded JPEG string
        suitable for Ollama's multimodal API, or None if unavailable.
        """
        if not self._available or self._cap is None:
            return None
        try:
            ret, frame = self._cap.read()
            if not ret:
                return None
            cv2 = self._cv2
            # Resize for faster inference
            frame = cv2.resize(frame, (320, 240))
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if not ok:
                return None
            return base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception as exc:
            logger.warning(f"Frame capture error: {exc}")
            return None

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
