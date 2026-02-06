import moderngl
import numpy as np
from PIL import Image, ImageFilter
from .base import BaseVisualizer


class CoverGlowVisualizer(BaseVisualizer):
    def __init__(self, ctx, resolution, assets, config):
        super().__init__(ctx, resolution, assets, config)

        self.tex_bg = self.ctx.texture((self.width, self.height), 4)

        cover_size = config.get("cover_size", 800)
        self.tex_fg = self.ctx.texture((cover_size, cover_size), 4)

        self.update_cover(assets["cover"])

        self.tex_bg.use(0)
        self.tex_fg.use(1)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex_bg;
                uniform sampler2D tex_fg;

                uniform float zoom;
                uniform vec2 shake;
                uniform float bloom;
                uniform vec2 resolution;
                uniform vec2 cover_res;

                in vec2 v_uv;
                out vec4 f_color;

                void main() {
                    vec4 bg_color = texture(tex_bg, v_uv);

                    vec2 center = vec2(0.5, 0.5);
                    vec2 uv = v_uv - shake;
                    uv = (uv - center) / zoom + center;

                    vec2 size_ratio = cover_res / resolution;
                    vec2 tex_uv = (uv - center) / size_ratio + center;

                    vec4 fg_color = vec4(0.0);
                    if (tex_uv.x > 0.0 && tex_uv.x < 1.0 && tex_uv.y > 0.0 && tex_uv.y < 1.0) {
                        fg_color = texture(tex_fg, tex_uv);
                        fg_color.rgb *= (1.0 + bloom * 0.5);
                    }

                    f_color = mix(bg_color, fg_color, fg_color.a);
                }
            """,
        )

        vertices = np.array(
            [-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1], dtype="f4"
        )
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_vert", "in_uv")]
        )

        self.prog["tex_bg"].value = 0
        self.prog["tex_fg"].value = 1
        self.prog["resolution"].value = tuple(resolution)
        self.prog["cover_res"].value = (cover_size, cover_size)

    def update_cover(self, cover_path):
        print(f"[CoverGlow] Switching to: {cover_path}")
        img = Image.open(cover_path).convert("RGBA")

        bg_img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        bg_img = bg_img.filter(
            ImageFilter.GaussianBlur(radius=self.config.get("blur_sigma", 30))
        )
        self.tex_bg.write(bg_img.tobytes())

        cover_size = self.config.get("cover_size", 800)
        fg_img = img.resize((cover_size, cover_size), Image.Resampling.LANCZOS)
        self.tex_fg.write(fg_img.tobytes())

    def render(self, audio_data, time):
        rms = audio_data.get("rms", 0.0)
        onset = audio_data.get("onset", 0.0)

        z_val = 1.0 + (rms * self.config.get("zoom_strength", 0.1))
        sx = (
            (np.random.random() - 0.5)
            * 0.01
            * onset
            * self.config.get("shake_strength", 5.0)
        )
        sy = (
            (np.random.random() - 0.5)
            * 0.01
            * onset
            * self.config.get("shake_strength", 5.0)
        )
        b_val = rms * self.config.get("bloom_strength", 1.0)

        self.prog["zoom"].value = z_val
        self.prog["shake"].value = (sx, sy)
        self.prog["bloom"].value = b_val

        self.vao.render(moderngl.TRIANGLE_STRIP)
