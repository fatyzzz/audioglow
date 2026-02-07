"""audioglow - GPU-accelerated audio-reactive music video generator."""

__version__ = "0.1.3"

from audioglow.api import render_video
from audioglow.result import RenderResult

__all__ = ["render_video", "RenderResult", "__version__"]
