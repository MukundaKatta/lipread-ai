"""Viseme vocabulary with 20+ viseme definitions."""

from __future__ import annotations

from lipread.models import Viseme


class VisemeVocabulary:
    """Defines the set of visemes used for visual speech recognition.

    A viseme is the visual equivalent of a phoneme -- the smallest
    distinguishable unit of lip/mouth shape during speech.
    """

    def __init__(self):
        self.visemes: list[Viseme] = self._build_vocabulary()
        self._id_map = {v.id: v for v in self.visemes}
        self._phoneme_map: dict[str, str] = {}
        for v in self.visemes:
            for p in v.phonemes:
                self._phoneme_map[p] = v.id

    def _build_vocabulary(self) -> list[Viseme]:
        """Build 21 viseme definitions covering English speech."""
        return [
            Viseme(id="V00", label="silence", description="Mouth at rest, lips closed",
                   phonemes=["SIL", "SP"], mouth_shape="closed"),
            Viseme(id="V01", label="bilabial_closed", description="Lips pressed together",
                   phonemes=["P", "B", "M"], mouth_shape="closed_pressed"),
            Viseme(id="V02", label="labiodental", description="Lower lip touches upper teeth",
                   phonemes=["F", "V"], mouth_shape="teeth_on_lip"),
            Viseme(id="V03", label="dental", description="Tongue between teeth",
                   phonemes=["TH", "DH"], mouth_shape="tongue_visible"),
            Viseme(id="V04", label="alveolar", description="Tongue tip at alveolar ridge",
                   phonemes=["T", "D", "N", "L"], mouth_shape="slightly_open"),
            Viseme(id="V05", label="alveolar_fricative", description="Narrow opening, teeth close",
                   phonemes=["S", "Z"], mouth_shape="narrow_slit"),
            Viseme(id="V06", label="postalveolar", description="Lips slightly rounded, teeth visible",
                   phonemes=["SH", "ZH", "CH", "JH"], mouth_shape="rounded_narrow"),
            Viseme(id="V07", label="velar", description="Mouth slightly open, back tongue raised",
                   phonemes=["K", "G", "NG"], mouth_shape="open_back"),
            Viseme(id="V08", label="glottal", description="Mouth open, throat constricted",
                   phonemes=["HH"], mouth_shape="open_relaxed"),
            Viseme(id="V09", label="approximant_r", description="Lips slightly rounded",
                   phonemes=["R"], mouth_shape="slight_round"),
            Viseme(id="V10", label="approximant_w", description="Lips rounded and protruded",
                   phonemes=["W"], mouth_shape="rounded_protruded"),
            Viseme(id="V11", label="palatal", description="Lips spread, tongue raised",
                   phonemes=["Y"], mouth_shape="spread"),
            Viseme(id="V12", label="vowel_open", description="Mouth wide open",
                   phonemes=["AA", "AH"], mouth_shape="wide_open"),
            Viseme(id="V13", label="vowel_mid_front", description="Mouth medium open, spread",
                   phonemes=["EH", "AE"], mouth_shape="mid_spread"),
            Viseme(id="V14", label="vowel_high_front", description="Mouth nearly closed, spread",
                   phonemes=["IH", "IY"], mouth_shape="narrow_spread"),
            Viseme(id="V15", label="vowel_mid_back", description="Mouth medium, slightly rounded",
                   phonemes=["AO", "OW"], mouth_shape="mid_round"),
            Viseme(id="V16", label="vowel_high_back", description="Lips rounded, small opening",
                   phonemes=["UH", "UW"], mouth_shape="small_round"),
            Viseme(id="V17", label="vowel_central", description="Mouth relaxed, neutral",
                   phonemes=["AX", "ER"], mouth_shape="neutral"),
            Viseme(id="V18", label="diphthong_ai", description="Open to spread transition",
                   phonemes=["AY", "EY"], mouth_shape="open_to_spread"),
            Viseme(id="V19", label="diphthong_ou", description="Open to rounded transition",
                   phonemes=["OY", "AW"], mouth_shape="open_to_round"),
            Viseme(id="V20", label="flap", description="Quick tongue tap",
                   phonemes=["DX"], mouth_shape="quick_open"),
        ]

    def get_viseme(self, viseme_id: str) -> Viseme | None:
        return self._id_map.get(viseme_id)

    def phoneme_to_viseme(self, phoneme: str) -> str:
        """Map a phoneme to its corresponding viseme ID."""
        return self._phoneme_map.get(phoneme.upper(), "V00")

    def size(self) -> int:
        return len(self.visemes)
