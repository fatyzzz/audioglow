"""Render result container."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RenderResult(os.PathLike):
    """Result of a render operation.

    Backward-compatible with ``Path`` — can be passed to ``open()``,
    ``Path()``, ``os.fspath()`` and any API that accepts path-like objects.

    Attributes:
        video_path: Path to the rendered video file.
        timestamps: List of track timestamps, each a dict with
            ``"title"`` (str) and ``"start_ms"`` (int).
    """

    video_path: Path
    timestamps: list[dict]

    def __fspath__(self) -> str:
        return str(self.video_path)

    def __str__(self) -> str:
        return str(self.video_path)

    def __repr__(self) -> str:
        return f"RenderResult({self.video_path!r}, {len(self.timestamps)} tracks)"
