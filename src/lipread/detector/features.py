"""Lip feature extractor computing lip shape and motion features."""

from __future__ import annotations

import math

import numpy as np

from lipread.models import LipPose


class LipFeatureExtractor:
    """Computes feature vectors from lip poses for the recognition model."""

    # Feature dimension: 20 landmarks * 2 coords + 4 shape metrics + 4 motion metrics = 48
    FEATURE_DIM = 48

    def extract_frame_features(self, pose: LipPose) -> np.ndarray:
        """Extract feature vector from a single lip pose.

        Returns a vector of shape (FEATURE_DIM,) containing:
        - 40 values: normalized (x, y) for each of 20 landmarks
        - 4 shape features: width, height, aspect ratio, openness
        - 4 motion features (zeros for single frame)
        """
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # Landmark coordinates (normalized to mouth center)
        if pose.landmarks:
            cx = np.mean([lm.x for lm in pose.landmarks])
            cy = np.mean([lm.y for lm in pose.landmarks])
            scale = max(pose.mouth_width, 1e-6)
            for i, lm in enumerate(pose.landmarks[:20]):
                features[i * 2] = (lm.x - cx) / scale
                features[i * 2 + 1] = (lm.y - cy) / scale

        # Shape features
        features[40] = pose.mouth_width / 100.0  # normalized
        features[41] = pose.mouth_height / 100.0
        features[42] = pose.aspect_ratio / 10.0
        features[43] = pose.openness

        # Motion features (filled by extract_sequence_features)
        # features[44:48] = 0

        return features

    def extract_sequence_features(self, poses: list[LipPose]) -> np.ndarray:
        """Extract feature matrix for a sequence of lip poses.

        Returns array of shape (num_frames, FEATURE_DIM) with motion features.
        """
        if not poses:
            return np.zeros((0, self.FEATURE_DIM), dtype=np.float32)

        features = np.stack([self.extract_frame_features(p) for p in poses])

        # Compute motion features (velocity of key metrics)
        for i in range(1, len(features)):
            features[i, 44] = features[i, 40] - features[i - 1, 40]  # width delta
            features[i, 45] = features[i, 41] - features[i - 1, 41]  # height delta
            features[i, 46] = features[i, 42] - features[i - 1, 42]  # aspect delta
            features[i, 47] = features[i, 43] - features[i - 1, 43]  # openness delta

        return features

    def compute_shape_similarity(self, pose_a: LipPose, pose_b: LipPose) -> float:
        """Compute similarity score (0-1) between two lip shapes."""
        fa = self.extract_frame_features(pose_a)
        fb = self.extract_frame_features(pose_b)
        # Cosine similarity on landmark features
        a = fa[:40]
        b = fb[:40]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
