"""Data models for lip reading pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class VideoFrame(BaseModel):
    """A single video frame with face/lip data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int
    timestamp_ms: float = 0.0
    width: int = 0
    height: int = 0
    pixels: Optional[np.ndarray] = None
    face_detected: bool = False
    lip_roi: Optional[tuple[int, int, int, int]] = None  # x, y, w, h


class LipLandmark(BaseModel):
    """A single lip landmark point."""

    index: int
    x: float
    y: float


class LipPose(BaseModel):
    """Lip shape described by 20 landmarks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int
    landmarks: list[LipLandmark] = Field(default_factory=list)
    mouth_width: float = 0.0
    mouth_height: float = 0.0
    aspect_ratio: float = 0.0
    openness: float = 0.0  # 0=closed, 1=fully open

    def compute_metrics(self) -> None:
        """Compute derived mouth metrics from landmarks."""
        if len(self.landmarks) < 4:
            return
        xs = [lm.x for lm in self.landmarks]
        ys = [lm.y for lm in self.landmarks]
        self.mouth_width = max(xs) - min(xs)
        self.mouth_height = max(ys) - min(ys)
        self.aspect_ratio = self.mouth_width / max(self.mouth_height, 1e-6)
        self.openness = min(self.mouth_height / max(self.mouth_width, 1e-6), 1.0)


class Viseme(BaseModel):
    """A visual phoneme - the visual representation of a speech sound."""

    id: str
    label: str
    description: str
    phonemes: list[str] = Field(default_factory=list)
    mouth_shape: str = ""  # compact description: open/closed/rounded etc.


class TranscriptionResult(BaseModel):
    """Result of lip reading transcription."""

    text: str = ""
    viseme_sequence: list[str] = Field(default_factory=list)
    phoneme_sequence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    frame_count: int = 0
    duration_ms: float = 0.0
    words: list[WordResult] = Field(default_factory=list)


class WordResult(BaseModel):
    """A single recognized word with timing."""

    word: str
    start_frame: int = 0
    end_frame: int = 0
    confidence: float = 0.0
    visemes: list[str] = Field(default_factory=list)
