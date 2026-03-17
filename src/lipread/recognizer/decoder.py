"""Viseme decoder mapping lip shapes to visemes to phonemes."""

from __future__ import annotations

from lipread.models import LipPose, TranscriptionResult, WordResult
from lipread.recognizer.vocabulary import VisemeVocabulary


class VisemeDecoder:
    """Decodes viseme sequences into phonemes and approximate text."""

    # Simple viseme-to-phoneme-to-character mapping for demonstration
    VISEME_TO_CHAR = {
        "V00": " ", "V01": "m", "V02": "f", "V03": "th",
        "V04": "t", "V05": "s", "V06": "sh", "V07": "k",
        "V08": "h", "V09": "r", "V10": "w", "V11": "y",
        "V12": "a", "V13": "e", "V14": "i", "V15": "o",
        "V16": "u", "V17": "uh", "V18": "ai", "V19": "ou",
        "V20": "d",
    }

    def __init__(self, vocabulary: VisemeVocabulary | None = None):
        self.vocabulary = vocabulary or VisemeVocabulary()

    def decode_indices(self, viseme_indices: list[int]) -> list[str]:
        """Convert viseme index sequence to viseme ID sequence."""
        viseme_ids = []
        for idx in viseme_indices:
            vid = f"V{idx:02d}"
            if self.vocabulary.get_viseme(vid):
                viseme_ids.append(vid)
            else:
                viseme_ids.append("V00")
        return viseme_ids

    def collapse_sequence(self, viseme_ids: list[str]) -> list[str]:
        """Remove consecutive duplicate visemes."""
        if not viseme_ids:
            return []
        collapsed = [viseme_ids[0]]
        for vid in viseme_ids[1:]:
            if vid != collapsed[-1]:
                collapsed.append(vid)
        return collapsed

    def visemes_to_phonemes(self, viseme_ids: list[str]) -> list[str]:
        """Map viseme IDs to their primary phonemes."""
        phonemes = []
        for vid in viseme_ids:
            viseme = self.vocabulary.get_viseme(vid)
            if viseme and viseme.phonemes:
                phonemes.append(viseme.phonemes[0])
            else:
                phonemes.append("SIL")
        return phonemes

    def decode(self, viseme_indices: list[int], frame_count: int = 0) -> TranscriptionResult:
        """Full decode pipeline: indices -> visemes -> phonemes -> text."""
        viseme_ids = self.decode_indices(viseme_indices)
        collapsed = self.collapse_sequence(viseme_ids)
        phonemes = self.visemes_to_phonemes(collapsed)

        # Build approximate text from viseme character mapping
        chars = []
        for vid in collapsed:
            char = self.VISEME_TO_CHAR.get(vid, "")
            if char:
                chars.append(char)
        text = "".join(chars).strip()

        # Segment into rough words on silence boundaries
        words = []
        current_word_chars: list[str] = []
        current_visemes: list[str] = []
        for vid in collapsed:
            if vid == "V00" and current_word_chars:
                words.append(WordResult(
                    word="".join(current_word_chars),
                    visemes=current_visemes,
                    confidence=0.5,
                ))
                current_word_chars = []
                current_visemes = []
            else:
                char = self.VISEME_TO_CHAR.get(vid, "")
                if char.strip():
                    current_word_chars.append(char)
                    current_visemes.append(vid)
        if current_word_chars:
            words.append(WordResult(
                word="".join(current_word_chars),
                visemes=current_visemes,
                confidence=0.5,
            ))

        return TranscriptionResult(
            text=text,
            viseme_sequence=collapsed,
            phoneme_sequence=phonemes,
            confidence=0.5,
            frame_count=frame_count,
            words=words,
        )
