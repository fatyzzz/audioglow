"""Public Python API for audioglow."""

from __future__ import annotations

from pathlib import Path

from audioglow.config import build_config


def render_video(
    tracks: list[str | Path],
    output: str | Path,
    video_type: str,
    *,
    resolution: tuple[int, int] | list[int] | None = None,
    fps: int | None = None,
    cover_size: int | None = None,
    zoom_strength: float | None = None,
    shake_strength: float | None = None,
    bloom_strength: float | None = None,
    blur_sigma: float | int | None = None,
    bitrate: str | None = None,
    crossfade_ms: int | None = None,
    overlay_path: str | Path | None = None,
    config_file: str | Path | None = None,
    tracks_count: int | None = None,
) -> Path:
    """Render an audio-reactive music video.

    Args:
        tracks: List of track directory paths. Each directory must contain
            a .wav audio file and a cover image (.png/.jpg/.jpeg).
        output: Output .mp4 file path.
        video_type: Visualizer type. One of: "cover_audio_reactive",
            "cover_overlay_green", "cover_overlay_spectrum", "cover_combo".
            Each type has its own tuned defaults for visual parameters.
        resolution: Video resolution as (width, height).
        fps: Frames per second.
        cover_size: Cover art size in pixels.
        zoom_strength: Audio-reactive zoom intensity.
        shake_strength: Audio-reactive shake intensity.
        bloom_strength: Audio-reactive bloom intensity.
        blur_sigma: Background blur radius.
        bitrate: Video bitrate string (e.g. "8M", "12M").
        crossfade_ms: Crossfade duration in ms between tracks.
        overlay_path: Path to overlay video file. Required for video types
            "cover_overlay_green" and "cover_combo".
        config_file: Optional JSON config file. Overrides type defaults;
            explicit kwargs override both.
        tracks_count: Max number of tracks to use from the list.

    Returns:
        Path to the rendered output .mp4 file.

    Raises:
        FileNotFoundError: If track directories or overlay video not found.
        ValueError: If config values are invalid.
        RuntimeError: If ffmpeg is not found or encoding fails.
    """
    track_dirs = [Path(t) for t in tracks]
    for td in track_dirs:
        if not td.exists():
            raise FileNotFoundError(f"Track directory not found: {td}")
        if not td.is_dir():
            raise ValueError(f"Track path is not a directory: {td}")

    output_path = Path(output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".mp4")

    kwargs: dict = {}
    for key, value in (
        ("resolution", resolution),
        ("fps", fps),
        ("cover_size", cover_size),
        ("zoom_strength", zoom_strength),
        ("shake_strength", shake_strength),
        ("bloom_strength", bloom_strength),
        ("blur_sigma", blur_sigma),
        ("bitrate", bitrate),
        ("crossfade_ms", crossfade_ms),
        ("overlay_path", str(overlay_path) if overlay_path is not None else None),
        ("tracks_count", tracks_count),
    ):
        if value is not None:
            kwargs[key] = value

    cfg = build_config(
        video_type,
        config_file=str(config_file) if config_file is not None else None,
        **kwargs,
    )

    tc = cfg.get("tracks_count")
    if tc is not None and tc < len(track_dirs):
        track_dirs = track_dirs[:tc]

    from audioglow.__main__ import _render_pipeline

    return _render_pipeline(cfg, track_dirs, output_path)
