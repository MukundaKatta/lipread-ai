"""Simulator for generating synthetic lip reading data."""

from __future__ import annotations

import math
import numpy as np

from lipread.models import VideoFrame, LipPose, TranscriptionResult
from lipread.detector.face import FaceDetector
from lipread.detector.lip import LipTracker
from lipread.detector.features import LipFeatureExtractor
from lipread.recognizer.vocabulary import VisemeVocabulary


class LipReadingSimulator:
    """Generates synthetic video frames and lip data for testing."""

    def __init__(self, fps: int = 30, width: int = 640, height: int = 480):
        self.fps = fps
        self.width = width
        self.height = height
        self.vocabulary = VisemeVocabulary()

    def generate_frames(self, duration_ms: float) -> list[VideoFrame]:
        """Generate synthetic video frames for a given duration."""
        num_frames = int(duration_ms / 1000.0 * self.fps)
        frames = []
        for i in range(num_frames):
            pixels = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
            frame = VideoFrame(
                frame_index=i,
                timestamp_ms=i * (1000.0 / self.fps),
                width=self.width,
                height=self.height,
                pixels=pixels,
            )
            frames.append(frame)
        return frames

    def generate_viseme_sequence(self, word: str) -> list[str]:
        """Generate a plausible viseme sequence for a word (simplified mapping)."""
        char_to_viseme = {
            "a": "V12", "b": "V01", "c": "V07", "d": "V04",
            "e": "V13", "f": "V02", "g": "V07", "h": "V08",
            "i": "V14", "j": "V06", "k": "V07", "l": "V04",
            "m": "V01", "n": "V04", "o": "V15", "p": "V01",
            "q": "V07", "r": "V09", "s": "V05", "t": "V04",
            "u": "V16", "v": "V02", "w": "V10", "x": "V07",
            "y": "V11", "z": "V05", " ": "V00",
        }
        visemes = []
        for ch in word.lower():
            vid = char_to_viseme.get(ch, "V00")
            # Don't repeat same viseme consecutively
            if not visemes or visemes[-1] != vid:
                visemes.append(vid)
        return visemes

    def simulate_pipeline(self, text: str, duration_ms: float = 2000.0) -> TranscriptionResult:
        """Run a full simulated lip reading pipeline."""
        frames = self.generate_frames(duration_ms)

        face_detector = FaceDetector()
        lip_tracker = LipTracker()
        feature_extractor = LipFeatureExtractor()

        detected_frames = face_detector.detect_batch(frames)
        poses = lip_tracker.track_sequence(detected_frames)
        features = feature_extractor.extract_sequence_features(poses)

        # Generate expected viseme sequence
        viseme_ids = self.generate_viseme_sequence(text)

        return TranscriptionResult(
            text=text,
            viseme_sequence=viseme_ids,
            phoneme_sequence=[],
            confidence=0.85,
            frame_count=len(frames),
            duration_ms=duration_ms,
        )
