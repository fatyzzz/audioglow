import json
from pathlib import Path


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("resolution", [1920, 1080])
    config.setdefault("fps", 30)
    config.setdefault("cover_size", 800)
    config.setdefault("zoom_strength", 0.1)
    config.setdefault("shake_strength", 5.0)
    config.setdefault("bloom_strength", 1.0)
    config.setdefault("blur_sigma", 30)
    config.setdefault("bitrate", "8M")

    return config
