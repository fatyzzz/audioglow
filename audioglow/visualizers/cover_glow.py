from .base import BaseVisualizer
from .shaders import (
    FRAG_COMMON_HEADER,
    GLSL_COMPOSITE_COVER,
    VERTEX_SHADER,
    build_fragment_shader,
)

COVER_GLOW_MAIN = """
void main() {
    vec4 bg = texture(tex_bg, v_uv);
    vec4 fg = compositeCover(v_uv);
    f_color = mix(bg, fg, fg.a);
}
"""


class CoverGlowVisualizer(BaseVisualizer):
    HAS_PALETTE = False
    HAS_OVERLAY = False
    HAS_SPECTRUM = False

    def __init__(self, ctx, resolution, assets, config):
        super().__init__(ctx, resolution, assets, config)
        self.prog = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=build_fragment_shader(
                FRAG_COMMON_HEADER,
                GLSL_COMPOSITE_COVER,
                COVER_GLOW_MAIN,
            ),
        )
        self._finish_init()
