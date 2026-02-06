import sys
import subprocess
import random
import argparse
from pathlib import Path
import threading
import queue

from audioglow.config import load_config
from audioglow.audio import AudioAnalyzer
from audioglow.engine import RenderEngine
from audioglow.mixer import AudioMixer
from audioglow.palette import extract_palette


def ffmpeg_writer(process, frame_queue):
    while True:
        frame_data = frame_queue.get()
        if frame_data is None:
            break
        try:
            process.stdin.write(frame_data)
        except (BrokenPipeError, OSError):
            break


def process_job(config_path: Path, tracks_dir: Path, output_dir: Path):
    try:
        config_id = config_path.parent.name
        category = config_path.parent.parent.name

        print(f"\n=== JOB: [{category}] - [{config_id}] ===")

        cfg = load_config(config_path)
        tracks_count = cfg.get("tracks_count", 1)

        if not tracks_dir.exists():
            print(f"[ERROR] Tracks directory not found: {tracks_dir}")
            return False

        all_track_dirs = sorted([d for d in tracks_dir.iterdir() if d.is_dir()])
        if not all_track_dirs:
            print(f"[ERROR] No track directories found in: {tracks_dir}")
            return False

        if len(all_track_dirs) < tracks_count:
            print(
                f"Warning: requested {tracks_count} tracks, but found only {len(all_track_dirs)}."
            )
            selected_tracks = all_track_dirs
        else:
            selected_tracks = random.sample(all_track_dirs, tracks_count)

        job_output = output_dir / category / config_id
        job_output.mkdir(parents=True, exist_ok=True)

        mixer = AudioMixer(cfg)
        mixed_audio, playlist = mixer.create_mix(
            selected_tracks, job_output, cfg["fps"]
        )
        mixer.save_timestamps(job_output)

        if len(selected_tracks) == 1:
            out_name = f"{selected_tracks[0].name}.mp4"
        else:
            out_name = f"MIX_{len(selected_tracks)}_tracks.mp4"
        output_file = job_output / out_name

        print(">>> Analyzing Full Mix...")
        analyzer = AudioAnalyzer(mixed_audio, cfg["fps"])
        audio_data = analyzer.analyze()
        total_frames = analyzer.total_frames

        print(">>> Pre-generating palettes for all tracks...")
        for track in playlist:
            track["palette"] = extract_palette(track["cover_path"])
            print(
                f"  Track @ frame {track['start_frame']}: palette = {track['palette'][:2]}..."
            )

        first_track = playlist[0]
        assets = {"cover": first_track["cover_path"], "audio": mixed_audio}
        engine = RenderEngine(cfg["resolution"], assets, cfg)

        engine.set_palette(first_track["palette"])

        next_tracks_queue = playlist[1:]

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{cfg['resolution'][0]}x{cfg['resolution'][1]}",
            "-pix_fmt", "rgba",
            "-r", str(cfg["fps"]),
            "-i", "-",
            "-i", str(mixed_audio),
            "-c:v", "h264_videotoolbox",
            "-b:v", cfg["bitrate"],
            "-allow_sw", "1",
            "-c:a", "aac",
            "-b:a", "320k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            str(output_file),
        ]

        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7
        )
        frame_queue = queue.Queue(maxsize=60)
        writer = threading.Thread(target=ffmpeg_writer, args=(process, frame_queue))
        writer.start()

        print(f"Rendering {total_frames} frames...")

        for i in range(total_frames):
            if next_tracks_queue:
                next_track = next_tracks_queue[0]
                if i >= next_track["start_frame"]:
                    engine.update_cover(next_track["cover_path"], next_track["palette"])
                    next_tracks_queue.pop(0)

            frame_payload = {
                "rms": audio_data["rms"][i],
                "onset": audio_data["onset"][i],
                "spectrum": (
                    audio_data["spectrum"][i] if "spectrum" in audio_data else None
                ),
            }

            t = i / cfg["fps"]

            buffer_ref = engine.render_frame_into_buffer(frame_payload, t)
            frame_queue.put(bytes(buffer_ref))

            if i % 120 == 0:
                perc = (i / total_frames) * 100
                sys.stdout.write(
                    f"\rRendering: {perc:.1f}% | Track queue: {len(next_tracks_queue)}"
                )
                sys.stdout.flush()

        frame_queue.put(None)
        writer.join()
        if process.stdin:
            process.stdin.close()
        process.wait()

        print(f"\n[DONE] Saved to: {output_file}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Job failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="audioglow - music video generator")
    parser.add_argument("--config", type=str, help="Path to a single config.json")
    parser.add_argument("--tracks-dir", type=str, help="Path to tracks directory")
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Process all configs in configs/")

    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent.resolve()

    if args.all:
        configs_root = project_dir / "configs"
        all_configs = sorted(list(configs_root.glob("*/*/config.json")))
        if not all_configs:
            print("No configs found.")
            return

        print(f"Found {len(all_configs)} jobs.")
        for cfg_path in all_configs:
            category = cfg_path.parent.parent.name
            tracks_dir = project_dir / "downloads" / category
            process_job(cfg_path, tracks_dir, Path(args.output))

    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            return

        if args.tracks_dir:
            tracks_dir = Path(args.tracks_dir)
        else:
            category = config_path.parent.parent.name
            tracks_dir = project_dir / "downloads" / category

        process_job(config_path, tracks_dir, Path(args.output))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
