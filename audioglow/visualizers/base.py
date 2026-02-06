import moderngl


class BaseVisualizer:
    def __init__(self, ctx: moderngl.Context, resolution: tuple, assets: dict, config: dict):
        self.ctx = ctx
        self.width, self.height = resolution
        self.assets = assets
        self.config = config

    def update_cover(self, cover_path):
        pass

    def render(self, audio_data, time: float):
        raise NotImplementedError("Each visualizer must implement render()")
