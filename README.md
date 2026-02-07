# audioglow

GPU-accelerated audio-reactive music video generator. Turns your tracks into visual content with OpenGL shaders — album covers react to beats, bass, and frequencies in real time.

## Installation

```bash
pip install audioglow
```

**Requirements:** `ffmpeg` must be installed and available in PATH.

## Quick Start

### Python API

```python
from audioglow import render_video

# Minimal — just tracks, output, and video type:
render_video(
    tracks=["tracks/my_song/"],
    output="output.mp4",
    video_type="cover_audio_reactive",
)

# Custom settings:
render_video(
    tracks=["tracks/song1/", "tracks/song2/"],
    output="mix.mp4",
    video_type="cover_audio_reactive",
    fps=60,
    resolution=(1920, 1080),
    bloom_strength=2.0,
    bitrate="16M",
    crossfade_ms=3000,
)
```

Each track directory must contain:

- One `.wav` audio file
- One cover image (`.png`, `.jpg`, or `.jpeg`)

### CLI

```bash
# Single config
audioglow --config config.json --tracks-dir ./tracks --output ./out

# Batch — all configs in configs/
audioglow --all --output ./out
```

## Video Types

| Type                     | Description                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| `cover_audio_reactive`   | Album cover with reactive zoom, shake, bloom on blurred background |
| `cover_overlay_green`    | Cover + green-screen video converted to palette-colored particles  |
| `cover_overlay_spectrum` | Cover + audio frequency spectrum wave visualization                |
| `cover_combo`            | All effects combined: particles + spectrum + cover                 |

Overlay types (`cover_overlay_green`, `cover_combo`) require `overlay_path` pointing to a green-screen video file.

## Parameters

| Parameter        | Default         | Description                                                      |
| ---------------- | --------------- | ---------------------------------------------------------------- |
| `resolution`     | `(1920, 1080)`  | Video resolution                                                 |
| `fps`            | `30`            | Frames per second                                                |
| `cover_size`     | `800`           | Cover art size (px)                                              |
| `zoom_strength`  | `0.3`           | Bass-reactive zoom intensity                                     |
| `shake_strength` | `4.0` / `0.5`\* | Beat-reactive camera shake                                       |
| `bloom_strength` | `1.5` / `3.0`\* | Glow intensity on beats                                          |
| `blur_sigma`     | `40` / `35`\*   | Background blur radius                                           |
| `bitrate`        | `"12M"`         | Video bitrate                                                    |
| `crossfade_ms`   | `4000`          | Crossfade between tracks (ms)                                    |
| `overlay_path`   | `None`          | Path to overlay video (required for overlay types)               |
| `config_file`    | `None`          | JSON config file (overrides type defaults, kwargs override both) |
| `tracks_count`   | all             | Max tracks to use                                                |

\*`cover_combo` type uses different defaults optimized for layered effects.

## Config Priority

```
Type Defaults  →  JSON config file  →  kwargs
   (lowest)                           (highest)
```

## JSON Config Example

```json
{
  "video_type": "cover_audio_reactive",
  "resolution": [1920, 1080],
  "fps": 30,
  "zoom_strength": 0.3,
  "shake_strength": 4.0,
  "bloom_strength": 1.5,
  "blur_sigma": 40,
  "bitrate": "12M",
  "tracks_count": 2,
  "crossfade_ms": 4000
}
```

## How It Works

1. **Audio analysis** — librosa extracts RMS energy, onset strength, and 64-band mel spectrogram
2. **Palette extraction** — dominant colors pulled from cover art
3. **GPU rendering** — ModernGL (OpenGL 3.3) fragment shaders render each frame
4. **Encoding** — ffmpeg encodes frames with hardware-accelerated H.264 (VideoToolbox / NVENC / VAAPI / QSV, fallback to libx264)

## License

MIT
