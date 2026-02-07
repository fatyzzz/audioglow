import moderngl

from audioglow.visualizers import get_visualizer_class


class RenderEngine:
    def __init__(self, resolution, assets, config):
        self.width, self.height = resolution
        self.ctx = moderngl.create_context(standalone=True)
        self.fbo = self.ctx.simple_framebuffer((self.width, self.height))
        self.fbo.use()
        self.frame_buffer = self.ctx.buffer(reserve=self.width * self.height * 4)

        viz_type = config.get("video_type", "cover_audio_reactive")
        print(f"Loading Visualizer: {viz_type}")

        VizClass = get_visualizer_class(viz_type)
        self.visualizer = VizClass(self.ctx, resolution, assets, config)

    def update_cover(self, new_cover_path, palette=None):
        if palette is not None:
            self.visualizer.set_palette(palette)
        self.visualizer.update_cover(new_cover_path)

    def set_palette(self, palette):
        self.visualizer.set_palette(palette)

    def render_frame_into_buffer(self, frame_data, time=0.0):
        self.fbo.clear()
        self.visualizer.render(frame_data, time)
        self.fbo.read_into(self.frame_buffer, components=4)
        return self.frame_buffer.read()

    def close(self):
        self.visualizer.close()
        self.frame_buffer.release()
        self.fbo.release()
        self.ctx.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
