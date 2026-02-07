import numpy as np

from .base import BaseVisualizer
from .shaders import (
    FRAG_COMMON_HEADER,
    GLSL_COMPOSITE_COVER,
    GLSL_GET_PALETTE_COLOR,
    GLSL_SPECTRUM_FUNCTIONS,
    VERTEX_SHADER,
    build_fragment_shader,
)

SPECTRUM_MAIN = """
void main() {
    vec4 bg = texture(tex_bg, v_uv);
    vec4 wave = drawWave(v_uv);
    vec4 layer1 = mix(bg, wave, wave.a);

    vec4 fg = compositeCover(v_uv);
    f_color = mix(layer1, fg, fg.a);
}
"""


class OverlaySpectrumVisualizer(BaseVisualizer):
    HAS_PALETTE = True
    HAS_OVERLAY = False
    HAS_SPECTRUM = True

    def __init__(self, ctx, resolution, assets, config):
        super().__init__(ctx, resolution, assets, config)
        self.prog = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=build_fragment_shader(
                FRAG_COMMON_HEADER,
                GLSL_GET_PALETTE_COLOR,
                GLSL_SPECTRUM_FUNCTIONS,
                GLSL_COMPOSITE_COVER,
                SPECTRUM_MAIN,
            ),
        )
        self._finish_init()

    def _pre_render(self, audio_data):
        spectrum = audio_data.get("spectrum")
        if spectrum is None:
            self.tex_spectrum.write(np.zeros(self.n_bands, dtype="f4").tobytes())
        else:
            self.tex_spectrum.write(spectrum.astype("f4").tobytes())
