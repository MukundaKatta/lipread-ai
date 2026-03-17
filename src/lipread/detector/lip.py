"""Lip tracker tracking lip movements across frames with 20 landmarks."""

from __future__ import annotations

import math

import numpy as np

from lipread.models import LipLandmark, LipPose, VideoFrame


# 20 lip landmark definitions (indices matching common face mesh subsets)
LIP_LANDMARK_NAMES = [
    "upper_outer_left", "upper_outer_center_left", "upper_outer_center",
    "upper_outer_center_right", "upper_outer_right",
    "lower_outer_right", "lower_outer_center_right", "lower_outer_center",
    "lower_outer_center_left", "lower_outer_left",
    "upper_inner_left", "upper_inner_center_left", "upper_inner_center",
    "upper_inner_center_right", "upper_inner_right",
    "lower_inner_right", "lower_inner_center_right", "lower_inner_center",
    "lower_inner_center_left", "lower_inner_left",
]


class LipTracker:
    """Tracks lip movements across video frames using 20 lip landmarks."""

    def __init__(self):
        self.history: list[LipPose] = []
        self._prev_landmarks: list[LipLandmark] | None = None

    def track(self, frame: VideoFrame) -> LipPose:
        """Extract lip pose from a single frame.

        Uses simulation when no real detector is available: generates
        landmarks based on the lip ROI with smooth motion.
        """
        roi = frame.lip_roi
        if roi is None:
            roi = (100, 200, 120, 40)

        x, y, w, h = roi
        cx, cy = x + w / 2, y + h / 2

        # Generate 20 landmarks in an elliptical lip shape
        landmarks = []
        for i in range(20):
            if i < 10:
                # Outer lip contour
                angle = math.pi * i / 9
                rx = w / 2
                ry = h / 2
            else:
                # Inner lip contour
                angle = math.pi * (i - 10) / 9
                rx = w / 3
                ry = h / 3

            lx = cx + rx * math.cos(angle) * (-1 if i < 5 or (10 <= i < 15) else 1)
            ly = cy - ry * math.sin(angle) if i < 10 else cy - ry * math.sin(angle) * 0.5

            # Add slight temporal smoothing with previous landmarks
            if self._prev_landmarks and i < len(self._prev_landmarks):
                alpha = 0.3
                lx = alpha * lx + (1 - alpha) * self._prev_landmarks[i].x
                ly = alpha * ly + (1 - alpha) * self._prev_landmarks[i].y

            landmarks.append(LipLandmark(index=i, x=lx, y=ly))

        pose = LipPose(frame_index=frame.frame_index, landmarks=landmarks)
        pose.compute_metrics()
        self._prev_landmarks = landmarks
        self.history.append(pose)
        return pose

    def track_sequence(self, frames: list[VideoFrame]) -> list[LipPose]:
        """Track lip poses across a sequence of frames."""
        return [self.track(f) for f in frames]

    def get_motion_delta(self, pose_a: LipPose, pose_b: LipPose) -> float:
        """Compute the average landmark displacement between two poses."""
        if not pose_a.landmarks or not pose_b.landmarks:
            return 0.0
        total = 0.0
        count = min(len(pose_a.landmarks), len(pose_b.landmarks))
        for i in range(count):
            dx = pose_a.landmarks[i].x - pose_b.landmarks[i].x
            dy = pose_a.landmarks[i].y - pose_b.landmarks[i].y
            total += math.sqrt(dx * dx + dy * dy)
        return total / max(count, 1)

    def reset(self) -> None:
        """Reset tracker state."""
        self.history.clear()
        self._prev_landmarks = None
