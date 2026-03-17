"""Tests for lipread-ai."""

import numpy as np
import pytest

from lipread.models import VideoFrame, LipLandmark, LipPose, Viseme, TranscriptionResult
from lipread.detector.face import FaceDetector
from lipread.detector.lip import LipTracker
from lipread.detector.features import LipFeatureExtractor
from lipread.recognizer.vocabulary import VisemeVocabulary
from lipread.recognizer.decoder import VisemeDecoder
from lipread.simulator import LipReadingSimulator


# --- Models ---

def test_lip_pose_compute_metrics():
    landmarks = [
        LipLandmark(index=0, x=10, y=20),
        LipLandmark(index=1, x=50, y=20),
        LipLandmark(index=2, x=30, y=10),
        LipLandmark(index=3, x=30, y=30),
    ]
    pose = LipPose(frame_index=0, landmarks=landmarks)
    pose.compute_metrics()
    assert pose.mouth_width == 40.0
    assert pose.mouth_height == 20.0
    assert pose.aspect_ratio == 2.0


def test_video_frame_creation():
    frame = VideoFrame(frame_index=0, width=640, height=480)
    assert frame.face_detected is False
    assert frame.lip_roi is None


# --- Face Detector ---

def test_face_detector():
    detector = FaceDetector()
    frame = VideoFrame(frame_index=0, width=640, height=480)
    result = detector.detect_face(frame)
    assert result.face_detected is True
    assert result.lip_roi is not None
    x, y, w, h = result.lip_roi
    assert w > 0 and h > 0


def test_face_detector_with_pixels():
    detector = FaceDetector()
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = VideoFrame(frame_index=0, width=640, height=480, pixels=pixels)
    lip_region = detector.extract_lip_region(frame)
    assert lip_region is not None
    assert lip_region.shape[2] == 3


def test_face_detector_batch():
    detector = FaceDetector()
    frames = [VideoFrame(frame_index=i, width=640, height=480) for i in range(5)]
    results = detector.detect_batch(frames)
    assert len(results) == 5
    assert all(f.face_detected for f in results)


# --- Lip Tracker ---

def test_lip_tracker_single():
    tracker = LipTracker()
    frame = VideoFrame(frame_index=0, lip_roi=(100, 200, 120, 40))
    pose = tracker.track(frame)
    assert len(pose.landmarks) == 20
    assert pose.mouth_width > 0


def test_lip_tracker_sequence():
    tracker = LipTracker()
    frames = [VideoFrame(frame_index=i, lip_roi=(100, 200, 120, 40)) for i in range(10)]
    poses = tracker.track_sequence(frames)
    assert len(poses) == 10
    assert len(tracker.history) == 10


def test_lip_tracker_motion_delta():
    tracker = LipTracker()
    f1 = VideoFrame(frame_index=0, lip_roi=(100, 200, 120, 40))
    f2 = VideoFrame(frame_index=1, lip_roi=(100, 200, 120, 60))
    p1 = tracker.track(f1)
    p2 = tracker.track(f2)
    delta = tracker.get_motion_delta(p1, p2)
    assert delta >= 0


# --- Feature Extractor ---

def test_feature_extractor_single():
    extractor = LipFeatureExtractor()
    landmarks = [LipLandmark(index=i, x=10 + i * 5, y=20 + i * 2) for i in range(20)]
    pose = LipPose(frame_index=0, landmarks=landmarks, mouth_width=50, mouth_height=20, aspect_ratio=2.5, openness=0.4)
    features = extractor.extract_frame_features(pose)
    assert features.shape == (48,)


def test_feature_extractor_sequence():
    extractor = LipFeatureExtractor()
    poses = []
    for i in range(5):
        landmarks = [LipLandmark(index=j, x=10 + j * 5 + i, y=20 + j * 2) for j in range(20)]
        poses.append(LipPose(frame_index=i, landmarks=landmarks, mouth_width=50, mouth_height=20))
    features = extractor.extract_sequence_features(poses)
    assert features.shape == (5, 48)


def test_shape_similarity():
    extractor = LipFeatureExtractor()
    landmarks = [LipLandmark(index=i, x=10 + i * 5, y=20 + i * 2) for i in range(20)]
    pose = LipPose(frame_index=0, landmarks=landmarks, mouth_width=50, mouth_height=20)
    sim = extractor.compute_shape_similarity(pose, pose)
    assert abs(sim - 1.0) < 0.01


# --- Vocabulary ---

def test_vocabulary_size():
    vocab = VisemeVocabulary()
    assert vocab.size() >= 20


def test_vocabulary_phoneme_mapping():
    vocab = VisemeVocabulary()
    assert vocab.phoneme_to_viseme("P") == "V01"
    assert vocab.phoneme_to_viseme("S") == "V05"
    assert vocab.phoneme_to_viseme("UNKNOWN") == "V00"


def test_vocabulary_get_viseme():
    vocab = VisemeVocabulary()
    v = vocab.get_viseme("V01")
    assert v is not None
    assert v.label == "bilabial_closed"


# --- Decoder ---

def test_decoder_indices():
    decoder = VisemeDecoder()
    ids = decoder.decode_indices([0, 1, 2, 5])
    assert ids == ["V00", "V01", "V02", "V05"]


def test_decoder_collapse():
    decoder = VisemeDecoder()
    collapsed = decoder.collapse_sequence(["V00", "V00", "V01", "V01", "V02"])
    assert collapsed == ["V00", "V01", "V02"]


def test_decoder_full():
    decoder = VisemeDecoder()
    result = decoder.decode([0, 0, 1, 1, 12, 12, 0, 4, 13], frame_count=9)
    assert isinstance(result, TranscriptionResult)
    assert len(result.text) > 0


# --- Model ---

def test_model_forward():
    from lipread.recognizer.model import LipReadingModel
    import torch

    model = LipReadingModel(input_dim=48, hidden_dim=32, num_layers=1, num_visemes=21)
    x = torch.randn(2, 10, 48)  # batch=2, seq=10, features=48
    out = model(x)
    assert out.shape == (2, 10, 21)


def test_model_predict():
    from lipread.recognizer.model import LipReadingModel

    model = LipReadingModel(input_dim=48, hidden_dim=32, num_layers=1, num_visemes=21)
    features = np.random.randn(10, 48).astype(np.float32)
    predictions = model.predict_visemes(features)
    assert len(predictions) == 10
    assert all(0 <= p < 21 for p in predictions)


# --- Simulator ---

def test_simulator_frames():
    sim = LipReadingSimulator(fps=10, width=320, height=240)
    frames = sim.generate_frames(1000)
    assert len(frames) == 10
    assert frames[0].pixels.shape == (240, 320, 3)


def test_simulator_viseme_sequence():
    sim = LipReadingSimulator()
    seq = sim.generate_viseme_sequence("hello")
    assert len(seq) > 0
    assert all(v.startswith("V") for v in seq)


def test_simulator_pipeline():
    sim = LipReadingSimulator(fps=10, width=160, height=120)
    result = sim.simulate_pipeline("hi", duration_ms=500)
    assert result.text == "hi"
    assert result.frame_count > 0
    assert result.confidence > 0
