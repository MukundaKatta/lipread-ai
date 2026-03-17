"""CNN+LSTM lip reading model for visual speech recognition."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from lipread.detector.features import LipFeatureExtractor


class LipReadingModel(nn.Module):
    """CNN+LSTM model for visual speech recognition.

    Architecture:
    1. Feature projection (linear layers acting on pre-extracted features)
    2. Temporal modeling (LSTM)
    3. Classification head (viseme prediction per frame)
    """

    def __init__(
        self,
        input_dim: int = LipFeatureExtractor.FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_visemes: int = 21,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_visemes = num_visemes

        # Feature projection (acts like CNN on feature vectors)
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_visemes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) feature tensor

        Returns:
            (batch, seq_len, num_visemes) logits
        """
        batch_size, seq_len, feat_dim = x.shape

        # Project features: reshape for BatchNorm1d
        x_flat = x.reshape(batch_size * seq_len, feat_dim)
        projected = self.feature_net(x_flat)
        projected = projected.reshape(batch_size, seq_len, -1)

        # Temporal modeling
        lstm_out, _ = self.lstm(projected)

        # Classify each frame
        out_flat = lstm_out.reshape(batch_size * seq_len, -1)
        logits = self.classifier(out_flat)
        logits = logits.reshape(batch_size, seq_len, self.num_visemes)

        return logits

    def predict_visemes(self, features: np.ndarray) -> list[int]:
        """Predict viseme indices from a feature sequence.

        Args:
            features: (seq_len, input_dim) numpy array

        Returns:
            List of predicted viseme indices
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0)  # add batch dim
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
            return predictions.squeeze(0).tolist()
