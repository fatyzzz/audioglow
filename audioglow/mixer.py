from pydub import AudioSegment
from pathlib import Path


class AudioMixer:
    def __init__(self, config):
        self.crossfade_ms = config.get("crossfade_ms", 4000)
        self.playlist_data = []

    def create_mix(self, track_dirs, output_dir, fps):
        print(f"--- MIXER: Processing {len(track_dirs)} tracks ---")

        combined = AudioSegment.empty()
        cursor_ms = 0
        self.playlist_data = []

        for i, track_dir in enumerate(track_dirs):
            try:
                audio_path = list(track_dir.glob("*.wav"))[0]
                cover_candidates = (
                    list(track_dir.glob("*.png"))
                    + list(track_dir.glob("*.jpg"))
                    + list(track_dir.glob("*.jpeg"))
                )
                cover_path = cover_candidates[0]
            except IndexError:
                print(f"[SKIP] Missing audio or cover in: {track_dir.name}")
                continue

            print(f"Adding: {track_dir.name}")

            sound = AudioSegment.from_wav(str(audio_path))

            if i == 0:
                combined += sound
                start_ms = 0
            else:
                start_ms = cursor_ms - (self.crossfade_ms // 2)
                if start_ms < 0:
                    start_ms = 0
                combined = combined.append(sound, crossfade=self.crossfade_ms)

            self.playlist_data.append(
                {
                    "title": track_dir.name,
                    "cover_path": cover_path,
                    "start_ms": start_ms,
                    "start_frame": int((start_ms / 1000.0) * fps),
                }
            )

            cursor_ms = len(combined)

        output_file = output_dir / "mix_master.wav"
        print(f"Exporting Mix to: {output_file}...")
        combined.export(str(output_file), format="wav")

        return output_file, self.playlist_data

    def save_timestamps(self, output_dir):
        txt_path = output_dir / "timestamps.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for item in self.playlist_data:
                ms = item["start_ms"]
                seconds = int((ms / 1000) % 60)
                minutes = int((ms / (1000 * 60)) % 60)
                hours = int((ms / (1000 * 60 * 60)) % 24)

                if hours > 0:
                    time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                else:
                    time_str = f"{minutes}:{seconds:02d}"

                f.write(f"{time_str} {item['title']}\n")

        print(f"Timestamps saved: {txt_path}")
