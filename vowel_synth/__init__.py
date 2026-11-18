"""本地共振峰元音合成演示器（声源/声道参数解耦的教学模型）。"""

from .analysis import (estimate_formant_peaks, estimate_harmonic_spacing,
                       harmonic_peaks, magnitude_spectrum,
                       measure_transfer_function, verify_formants)
from .audio_io import (RealtimePlayer, read_wav, sounddevice_available,
                       write_wav)
from .engine import RealtimeEngine, SynthParams, render
from .filters import FormantCascade, design_resonator
from .source import GlottalSource
from .vowels import get_vowel, list_ascii, list_symbols, vowel_table

__all__ = [
    "SynthParams", "render", "RealtimeEngine",
    "FormantCascade", "design_resonator", "GlottalSource",
    "get_vowel", "list_symbols", "list_ascii", "vowel_table",
    "magnitude_spectrum", "estimate_harmonic_spacing",
    "estimate_formant_peaks", "harmonic_peaks",
    "measure_transfer_function", "verify_formants",
    "write_wav", "read_wav", "RealtimePlayer", "sounddevice_available",
]
