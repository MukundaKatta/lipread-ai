"""Face detector for face and lip ROI extraction."""

from __future__ import annotations

import numpy as np

from lipread.models import VideoFrame


class FaceDetector:
    """Detects faces and extracts lip region of interest from video frames.

    In production, this would wrap a face detection model (e.g., dlib, mediapipe).
    This implementation provides a simulation-capable interface.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._detection_count = 0

    def detect_face(self, frame: VideoFrame) -> VideoFrame:
        """Detect face in frame and set face_detected flag and lip ROI.

        For simulation, assumes face is centered in the frame occupying ~60%
        of the area, with lip region in the lower third of the face.
        """
        if frame.pixels is not None:
            h, w = frame.pixels.shape[:2]
        else:
            w, h = frame.width or 640, frame.height or 480

        # Simulated face bounding box (center 60%)
        face_x = int(w * 0.2)
        face_y = int(h * 0.15)
        face_w = int(w * 0.6)
        face_h = int(h * 0.7)

        # Lip ROI: lower portion of face region
        lip_x = face_x + int(face_w * 0.2)
        lip_y = face_y + int(face_h * 0.65)
        lip_w = int(face_w * 0.6)
        lip_h = int(face_h * 0.2)

        frame.face_detected = True
        frame.lip_roi = (lip_x, lip_y, lip_w, lip_h)
        self._detection_count += 1
        return frame

    def extract_lip_region(self, frame: VideoFrame) -> np.ndarray | None:
        """Extract the lip ROI pixels from a frame."""
        if not frame.face_detected or frame.lip_roi is None:
            frame = self.detect_face(frame)

        if frame.pixels is None or frame.lip_roi is None:
            return None

        x, y, w, h = frame.lip_roi
        return frame.pixels[y : y + h, x : x + w].copy()

    def detect_batch(self, frames: list[VideoFrame]) -> list[VideoFrame]:
        """Detect faces in a batch of frames."""
        return [self.detect_face(f) for f in frames]
