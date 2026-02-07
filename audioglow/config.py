from __future__ import annotations

import json
from pathlib import Path

_BASE_DEFAULTS = {
    "resolution": [1920, 1080],
    "fps": 30,
    "cover_size": 800,
    "zoom_strength": 0.3,
    "shake_strength": 4.0,
    "bloom_strength": 1.5,
    "blur_sigma": 40,
    "bitrate": "12M",
    "crossfade_ms": 4000,
}

TYPE_DEFAULTS: dict[str, dict] = {
    "cover_audio_reactive": {**_BASE_DEFAULTS},
    "cover_overlay_green": {**_BASE_DEFAULTS},
    "cover_overlay_spectrum": {**_BASE_DEFAULTS},
    "cover_combo": {
        **_BASE_DEFAULTS,
        "shake_strength": 0.5,
        "bloom_strength": 3.0,
        "blur_sigma": 35,
    },
}

VALID_VIDEO_TYPES = set(TYPE_DEFAULTS)

_OVERLAY_TYPES = {"cover_overlay_green", "cover_combo"}


def build_config(
    video_type: str, config_file: Path | str | None = None, **kwargs
) -> dict:
    """Build a validated config dict.

    Priority: TYPE_DEFAULTS -> config_file JSON -> kwargs.
    """
    if video_type not in VALID_VIDEO_TYPES:
        raise ValueError(
            f"Unknown video_type: '{video_type}'. "
            f"Must be one of: {sorted(VALID_VIDEO_TYPES)}"
        )

    cfg = dict(TYPE_DEFAULTS[video_type])
    cfg["video_type"] = video_type

    if config_file is not None:
        cfg.update(_load_json(Path(config_file)))

    for key, value in kwargs.items():
        if value is not None:
            if key == "resolution" and isinstance(value, tuple):
                value = list(value)
            cfg[key] = value

    _validate(cfg)
    return cfg


def load_config(path: Path) -> dict:
    """Load config from a JSON file. Legacy API for CLI."""
    file_cfg = _load_json(path)
    video_type = file_cfg.pop("video_type", "cover_audio_reactive")
    return build_config(video_type, **file_cfg)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate(config):
    errors = []

    res = config.get("resolution", [1920, 1080])
    if (
        not isinstance(res, list)
        or len(res) != 2
        or not all(isinstance(x, int) and x > 0 for x in res)
    ):
        errors.append(
            f"resolution must be [width, height] with positive ints, got {res}"
        )

    fps = config.get("fps", 30)
    if not isinstance(fps, (int, float)) or fps <= 0:
        errors.append(f"fps must be positive number, got {fps}")

    vt = config.get("video_type", "cover_audio_reactive")
    if vt not in VALID_VIDEO_TYPES:
        errors.append(
            f"video_type must be one of {sorted(VALID_VIDEO_TYPES)}, got '{vt}'"
        )

    if vt in _OVERLAY_TYPES:
        overlay_path = config.get("overlay_path")
        if not overlay_path:
            errors.append(
                f"video_type '{vt}' requires 'overlay_path'. "
                f"Provide a path to an overlay video file."
            )
        elif not Path(overlay_path).exists():
            errors.append(f"overlay_path not found: {overlay_path}")

    for key in ("zoom_strength", "shake_strength", "bloom_strength", "blur_sigma"):
        val = config.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val < 0):
            errors.append(f"{key} must be non-negative number, got {val}")

    cs = config.get("cover_size", 800)
    if not isinstance(cs, int) or cs <= 0:
        errors.append(f"cover_size must be positive int, got {cs}")

    cf = config.get("crossfade_ms", 4000)
    if not isinstance(cf, int) or cf < 0:
        errors.append(f"crossfade_ms must be non-negative int, got {cf}")

    tc = config.get("tracks_count")
    if tc is not None and (not isinstance(tc, int) or tc < 1):
        errors.append(f"tracks_count must be positive int, got {tc}")

    br = config.get("bitrate", "8M")
    if not isinstance(br, str) or not br:
        errors.append(f"bitrate must be a non-empty string, got {br}")

    if errors:
        raise ValueError("Invalid config:\n" + "\n".join(f"  - {e}" for e in errors))
