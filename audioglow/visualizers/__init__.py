from .cover_glow import CoverGlowVisualizer
from .overlay_green import OverlayGreenVisualizer
from .overlay_spectrum import OverlaySpectrumVisualizer
from .combo import ComboVisualizer

VISUALIZERS = {
    "cover_audio_reactive": CoverGlowVisualizer,
    "cover_overlay_green": OverlayGreenVisualizer,
    "cover_overlay_spectrum": OverlaySpectrumVisualizer,
    "cover_combo": ComboVisualizer,
}


def get_visualizer_class(name):
    if name not in VISUALIZERS:
        return CoverGlowVisualizer
    return VISUALIZERS[name]
