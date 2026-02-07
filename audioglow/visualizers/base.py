import queue
import threading
import time as _time
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageFilter


class BaseVisualizer:
    """Base class handling cover textures, quad geometry, zoom/shake/bloom,
    palette texture, and optional video overlay loading.

    Subclasses set class-level flags to opt into features,
    build self.prog (shader program), then call self._finish_init().
    """

    HAS_PALETTE = False
    HAS_OVERLAY = False
    HAS_SPECTRUM = False

    def __init__(self, ctx, resolution, assets, config):
        self.ctx = ctx
        self.width, self.height = resolution
        self.assets = assets
        self.config = config
        self._rng = np.random.RandomState(42)
        self.prog: moderngl.Program = None  # type: ignore[assignment]

        self.cover_size = config.get("cover_size", 800)

        self.tex_bg = ctx.texture((self.width, self.height), 4)
        self.tex_fg = ctx.texture((self.cover_size, self.cover_size), 4)

        if self.HAS_PALETTE:
            self.tex_palette = ctx.texture((5, 1), 3, dtype="f4")
            self.palette = [(0.5, 0.5, 0.5)] * 5

        if self.HAS_SPECTRUM:
            self.n_bands = 64
            self.tex_spectrum = ctx.texture((self.n_bands, 1), 1, dtype="f4")
            self.tex_spectrum.write(np.zeros(self.n_bands, dtype="f4").tobytes())

        if self.HAS_OVERLAY:
            self._init_overlay(config)

        self._load_cover_textures(assets["cover"])

    def _finish_init(self):
        """Called by subclass after self.prog is created.
        Sets up VBO/VAO, binds textures, sets common uniforms."""
        vertices = np.array(
            [-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1],
            dtype="f4",
        )
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_vert", "in_uv")]
        )
        self._bind_textures()
        self._set_common_uniforms()

    def _bind_textures(self):
        unit = 0
        self.tex_bg.use(unit)
        self.prog["tex_bg"].value = unit
        unit += 1

        self.tex_fg.use(unit)
        self.prog["tex_fg"].value = unit
        unit += 1

        if self.HAS_OVERLAY:
            self.tex_overlay.use(unit)
            self.prog["tex_overlay"].value = unit
            unit += 1

        if self.HAS_SPECTRUM:
            self.tex_spectrum.use(unit)
            self.prog["tex_spectrum"].value = unit
            unit += 1

        if self.HAS_PALETTE:
            self.tex_palette.use(unit)
            self.prog["tex_palette"].value = unit
            unit += 1

    def _set_common_uniforms(self):
        self.prog["resolution"].value = (self.width, self.height)
        self.prog["cover_res"].value = (self.cover_size, self.cover_size)
        if self.HAS_PALETTE:
            self._update_palette_texture()

    # ---- Cover loading ----

    def _load_cover_textures(self, cover_path):
        img = Image.open(cover_path).convert("RGBA")
        bg = img.resize((self.width, self.height), Image.Resampling.LANCZOS).filter(
            ImageFilter.GaussianBlur(radius=self.config.get("blur_sigma", 30))
        )
        self.tex_bg.write(bg.tobytes())

        fg = img.resize((self.cover_size, self.cover_size), Image.Resampling.LANCZOS)
        self.tex_fg.write(fg.tobytes())

    def update_cover(self, cover_path):
        print(f"[{self.__class__.__name__}] Switching to: {Path(cover_path).name}")
        self._load_cover_textures(cover_path)
        if self.HAS_PALETTE:
            self._update_palette_texture()
            print(f"[{self.__class__.__name__}] Palette applied: {self.palette[:2]}...")

    # ---- Palette ----

    def set_palette(self, palette):
        if self.HAS_PALETTE:
            self.palette = palette
            self._update_palette_texture()
            print(f"[{self.__class__.__name__}] Palette updated: {palette[:2]}...")

    def _update_palette_texture(self):
        if not self.HAS_PALETTE:
            return
        data = np.array([list(c) for c in self.palette], dtype="f4")
        self.tex_palette.write(data.tobytes())

    # ---- Overlay video ----

    def _init_overlay(self, config):
        import cv2

        overlay_path = config.get("overlay_path")
        if not overlay_path:
            raise FileNotFoundError(
                "overlay_path is required for this video type. "
                "Provide a path to an overlay video file."
            )
        self.overlay_path = Path(overlay_path)
        if not self.overlay_path.exists():
            raise FileNotFoundError(f"Overlay not found: {self.overlay_path}")

        cap = cv2.VideoCapture(str(self.overlay_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open overlay: {self.overlay_path}")
        self.ov_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ov_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.tex_overlay = self.ctx.texture((self.ov_w, self.ov_h), 3)
        self._fill_overlay_key_green()

        self._overlay_queue = queue.Queue(maxsize=128)
        self._overlay_stop = threading.Event()
        self._overlay_thread = threading.Thread(target=self._video_loader, daemon=True)
        self._overlay_thread.start()

    def _fill_overlay_key_green(self):
        key = np.zeros((self.ov_h, self.ov_w, 3), dtype=np.uint8)
        key[..., 0] = 82
        key[..., 1] = 178
        key[..., 2] = 2
        self.tex_overlay.write(key.tobytes())

    def _video_loader(self):
        import cv2  # already validated in _init_overlay

        cap = cv2.VideoCapture(str(self.overlay_path))
        if not cap.isOpened():
            return
        while not self._overlay_stop.is_set():
            if self._overlay_queue.full():
                _time.sleep(0.01)
                continue
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._overlay_queue.put(rgb.tobytes())
        cap.release()

    def _pump_overlay_frame(self):
        try:
            data = self._overlay_queue.get_nowait()
            self.tex_overlay.write(data)
        except queue.Empty:
            pass

    # ---- Zoom / shake / bloom ----

    def _apply_effects(self, rms, onset):
        z = 1.0 + rms * self.config.get("zoom_strength", 0.1)
        strength = self.config.get("shake_strength", 5.0)
        sx = (self._rng.random() - 0.5) * 0.01 * onset * strength
        sy = (self._rng.random() - 0.5) * 0.01 * onset * strength
        b = rms * self.config.get("bloom_strength", 1.0)

        self.prog["zoom"].value = z
        self.prog["shake"].value = (sx, sy)
        self.prog["bloom"].value = b

    # ---- Render ----

    def render(self, audio_data, time):
        rms = audio_data.get("rms", 0.0)
        onset = audio_data.get("onset", 0.0)
        self._pre_render(audio_data)
        self._apply_effects(rms, onset)
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def _pre_render(self, audio_data):
        """Hook for subclasses to update textures before draw."""
        pass

    # ---- Resource cleanup ----

    def close(self):
        if self.HAS_OVERLAY and hasattr(self, "_overlay_stop"):
            self._overlay_stop.set()
            self._overlay_thread.join(timeout=2.0)
        for attr in (
            "tex_bg",
            "tex_fg",
            "tex_palette",
            "tex_spectrum",
            "tex_overlay",
            "vbo",
            "prog",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
