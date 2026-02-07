from .base import BaseVisualizer
from .shaders import (
    FRAG_COMMON_HEADER,
    GLSL_COMPOSITE_COVER,
    GLSL_GET_PALETTE_COLOR,
    GLSL_GET_PARTICLE_COLOR,
    VERTEX_SHADER,
    build_fragment_shader,
)

OVERLAY_GREEN_MAIN = """
void main() {
    vec4 bg = texture(tex_bg, v_uv);

    vec4 ov_raw = texture(tex_overlay, v_uv);
    vec4 particles = getParticleColor(ov_raw, v_uv);
    vec4 layer1 = mix(bg, particles, particles.a);

    vec4 fg = compositeCover(v_uv);
    f_color = mix(layer1, fg, fg.a);
}
"""


class OverlayGreenVisualizer(BaseVisualizer):
    HAS_PALETTE = True
    HAS_OVERLAY = True
    HAS_SPECTRUM = False

    def __init__(self, ctx, resolution, assets, config):
        super().__init__(ctx, resolution, assets, config)
        self.prog = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=build_fragment_shader(
                FRAG_COMMON_HEADER,
                GLSL_GET_PALETTE_COLOR,
                GLSL_GET_PARTICLE_COLOR,
                GLSL_COMPOSITE_COVER,
                OVERLAY_GREEN_MAIN,
            ),
        )
        self._finish_init()

    def _pre_render(self, audio_data):
        self._pump_overlay_frame()
