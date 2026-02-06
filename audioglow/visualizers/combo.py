import moderngl
import numpy as np
import cv2
import threading
import queue
from pathlib import Path
from PIL import Image, ImageFilter
from .base import BaseVisualizer
import time


class ComboVisualizer(BaseVisualizer):
    def __init__(self, ctx, resolution, assets, config):
        super().__init__(ctx, resolution, assets, config)

        project_root = Path(__file__).resolve().parent.parent.parent
        default_overlay = project_root / "overlays" / "particle.mp4"
        overlay_path_str = config.get("overlay_path", str(default_overlay))
        self.overlay_path = Path(overlay_path_str)
        if not self.overlay_path.exists():
            raise FileNotFoundError(f"Overlay not found: {self.overlay_path}")

        self.tex_bg = self.ctx.texture((self.width, self.height), 4)

        cover_size = config.get("cover_size", 800)
        self.tex_fg = self.ctx.texture((cover_size, cover_size), 4)

        self.n_bands = 64
        self.tex_spectrum = self.ctx.texture((self.n_bands, 1), 1, dtype="f4")
        self.tex_spectrum.write(np.zeros((self.n_bands,), dtype="f4").tobytes())

        self.tex_palette = self.ctx.texture((5, 1), 3, dtype="f4")
        self.palette = [(0.5, 0.5, 0.5)] * 5

        cap = cv2.VideoCapture(str(self.overlay_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open overlay video: {self.overlay_path}")
        ov_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ov_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.ov_w, self.ov_h = ov_w, ov_h
        self.tex_overlay = self.ctx.texture((ov_w, ov_h), 3)

        self._init_overlay_texture_key_green()

        img = Image.open(assets["cover"]).convert("RGBA")
        bg_img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        bg_img = bg_img.filter(
            ImageFilter.GaussianBlur(radius=config.get("blur_sigma", 30))
        )
        self.tex_bg.write(bg_img.tobytes())

        fg_img = img.resize((cover_size, cover_size), Image.Resampling.LANCZOS)
        self.tex_fg.write(fg_img.tobytes())

        self.tex_bg.use(0)
        self.tex_fg.use(1)
        self.tex_overlay.use(2)
        self.tex_spectrum.use(3)
        self.tex_palette.use(4)

        self.frame_queue = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()
        self.loader_thread = threading.Thread(target=self._video_loader, daemon=True)
        self.loader_thread.start()

        self.prog = self.ctx.program(
            vertex_shader=r"""
                #version 330
                in vec2 in_vert;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_uv = in_uv;
                }
            """,
            fragment_shader=r"""
                #version 330
                uniform sampler2D tex_bg;
                uniform sampler2D tex_fg;
                uniform sampler2D tex_overlay;
                uniform sampler2D tex_spectrum;
                uniform sampler2D tex_palette;

                uniform float zoom;
                uniform vec2 shake;
                uniform float bloom;
                uniform vec2 resolution;
                uniform vec2 cover_res;

                in vec2 v_uv;
                out vec4 f_color;

                vec3 getPaletteColor(int index) {
                    float u = (float(index) + 0.5) / 5.0;
                    return texture(tex_palette, vec2(u, 0.5)).rgb;
                }

                vec4 getParticleColor(vec4 color, vec2 uv) {
                    vec3 target = vec3(0.321, 0.698, 0.008);
                    float dist = distance(color.rgb, target);

                    float threshold = 0.28;
                    float softness  = 0.05;
                    float key = smoothstep(threshold - softness, threshold + softness, dist);

                    float luma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                    float luma_mask = smoothstep(0.03, 0.10, luma);

                    float pattern = (uv.y * 3.0 + uv.x * 2.0);
                    int idx = clamp(int(mod(pattern, 5.0)), 0, 4);
                    vec3 base_color = getPaletteColor(idx);

                    float intensity = clamp(luma * 1.6, 0.0, 1.0);
                    vec3 rgb = base_color * intensity;

                    float alpha = key * luma_mask;
                    return vec4(rgb, alpha);
                }

                float getVal(int index) {
                    index = clamp(index, 0, 63);
                    float u = (float(index) + 0.5) / 64.0;
                    return texture(tex_spectrum, vec2(u, 0.5)).r;
                }

                float getBSplineAmp(float t) {
                    float size = 64.0;
                    float p = t * size;
                    int i = int(floor(p));
                    float f = fract(p);
                    float p0 = getVal(i - 1); float p1 = getVal(i);
                    float p2 = getVal(i + 1); float p3 = getVal(i + 2);
                    float b0 = (1.0 - f)*(1.0 - f)*(1.0 - f)/6.0;
                    float b1 = (3.0*f*f*f - 6.0*f*f + 4.0)/6.0;
                    float b2 = (-3.0*f*f*f + 3.0*f*f + 3.0*f + 1.0)/6.0;
                    float b3 = f*f*f/6.0;
                    return p0*b0 + p1*b1 + p2*b2 + p3*b3;
                }

                vec4 drawWave(vec2 uv) {
                    float x_screen = uv.x;
                    float x_sampled = 0.04 + x_screen * 0.74;
                    float amplitude = getBSplineAmp(x_sampled);

                    amplitude = pow(amplitude, 2.0);
                    amplitude = max(0.0, amplitude - 0.05);
                    amplitude *= 1.7;

                    float fade_mask = 1.0 - smoothstep(0.85, 1.0, x_screen);
                    amplitude *= fade_mask;

                    float max_h = 0.17;
                    float wave_h = max_h * (1.0 - exp(-amplitude / max_h));
                    float y_rel = 1.0 - uv.y;

                    float glow_size = 0.03;
                    float edge_sharp = 0.002;
                    float stroke_size = 0.003;

                    float mask_fill = 1.0 - smoothstep(wave_h, wave_h + edge_sharp, y_rel);
                    float glow_mask = 1.0 - smoothstep(wave_h, wave_h + glow_size, y_rel);
                    float glow_only = clamp(glow_mask - mask_fill, 0.0, 1.0);

                    float stroke_top = wave_h + stroke_size;
                    float mask_stroke = 1.0 - smoothstep(stroke_top, stroke_top + edge_sharp, y_rel);
                    float outline = clamp(mask_stroke - mask_fill, 0.0, 1.0);

                    int c_idx = clamp(int(x_screen * 4.0), 0, 4);
                    vec3 c1 = getPaletteColor(c_idx);
                    vec3 c2 = getPaletteColor(min(c_idx + 1, 4));
                    vec3 col_fill = mix(c1, c2, fract(x_screen * 4.0));
                    vec3 col_stroke = mix(col_fill, vec3(1.0), 0.35);

                    vec4 res = vec4(0.0);
                    res.rgb += col_fill * mask_fill;
                    res.a = max(res.a, mask_fill * 0.95);
                    res.rgb = mix(res.rgb, col_stroke, outline);
                    res.a = max(res.a, outline);
                    res.rgb += col_fill * glow_only * 0.6;
                    res.a = max(res.a, glow_only * 0.5);
                    return res;
                }

                void main() {
                    vec4 bg = texture(tex_bg, v_uv);

                    vec4 ov_raw = texture(tex_overlay, v_uv);
                    vec4 particles = getParticleColor(ov_raw, v_uv);
                    vec4 layer1 = mix(bg, particles, particles.a);

                    vec4 wave = drawWave(v_uv);
                    vec4 layer2 = mix(layer1, wave, wave.a);

                    vec2 center = vec2(0.5);
                    vec2 uv = v_uv - shake;
                    uv = (uv - center) / zoom + center;

                    vec2 size_ratio = cover_res / resolution;
                    vec2 tex_uv = (uv - center) / size_ratio + center;

                    vec4 fg = vec4(0.0);
                    if (tex_uv.x > 0.0 && tex_uv.x < 1.0 && tex_uv.y > 0.0 && tex_uv.y < 1.0) {
                        fg = texture(tex_fg, tex_uv);
                        fg.rgb *= (1.0 + bloom * 0.5);
                    }

                    f_color = mix(layer2, fg, fg.a);
                }
            """,
        )

        self._update_uniforms()

        vertices = np.array(
            [-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1],
            dtype="f4",
        )
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_vert", "in_uv")]
        )

    def _init_overlay_texture_key_green(self):
        key_green = np.zeros((self.ov_h, self.ov_w, 3), dtype=np.uint8)
        key_green[..., 0] = 82
        key_green[..., 1] = 178
        key_green[..., 2] = 2
        self.tex_overlay.write(key_green.tobytes())

    def update_cover(self, cover_path):
        print(f"[Combo] Switching assets: {Path(cover_path).name}")
        img = Image.open(cover_path).convert("RGBA")

        bg_img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        bg_img = bg_img.filter(
            ImageFilter.GaussianBlur(radius=self.config.get("blur_sigma", 30))
        )
        self.tex_bg.write(bg_img.tobytes())

        cover_size = self.config.get("cover_size", 800)
        fg_img = img.resize((cover_size, cover_size), Image.Resampling.LANCZOS)
        self.tex_fg.write(fg_img.tobytes())

        if hasattr(self, "tex_palette"):
            self._update_palette_texture()
            print(f"[Combo] Palette applied: {self.palette[:2]}...")

    def _update_uniforms(self):
        self.prog["tex_bg"].value = 0
        self.prog["tex_fg"].value = 1
        self.prog["tex_overlay"].value = 2
        self.prog["tex_spectrum"].value = 3
        self.prog["tex_palette"].value = 4
        self.prog["resolution"].value = (self.width, self.height)
        cover_size = self.config.get("cover_size", 800)
        self.prog["cover_res"].value = (cover_size, cover_size)

        self._update_palette_texture()

    def _update_palette_texture(self):
        palette_data = np.array([list(c) for c in self.palette], dtype="f4")
        self.tex_palette.write(palette_data.tobytes())

    def _video_loader(self):
        cap = cv2.VideoCapture(str(self.overlay_path))
        if not cap.isOpened():
            return
        while not self.stop_event.is_set():
            if self.frame_queue.full():
                time.sleep(0.01)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_queue.put(frame_rgb.tobytes())
        cap.release()

    def render(self, audio_data, time):
        try:
            frame_bytes = self.frame_queue.get_nowait()
            self.tex_overlay.write(frame_bytes)
        except queue.Empty:
            pass

        spectrum = audio_data.get("spectrum")
        if spectrum is None:
            zeros = np.zeros((64,), dtype="f4")
            self.tex_spectrum.write(zeros.tobytes())
        else:
            self.tex_spectrum.write(spectrum.astype("f4").tobytes())

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

    def __del__(self):
        self.stop_event.set()
        if hasattr(self, "loader_thread"):
            self.loader_thread.join(timeout=1.0)
