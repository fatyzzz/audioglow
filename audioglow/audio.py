import librosa
import numpy as np
from scipy.interpolate import interp1d


class AudioAnalyzer:
    def __init__(self, audio_path, fps):
        self.audio_path = str(audio_path)
        self.fps = fps
        self.y = None
        self.sr = None
        self.duration = 0
        self.total_frames = 0

    def analyze(self):
        print(f"Loading audio: {self.audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=None)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.total_frames = int(self.duration * self.fps)

        hop_length = 512
        rms = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=hop_length)[0]
        onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=hop_length
        )

        times_orig = librosa.frames_to_time(
            np.arange(len(rms)), sr=self.sr, hop_length=hop_length
        )
        times_target = np.linspace(0, self.duration, self.total_frames)

        f_rms = interp1d(
            times_orig, self._normalize(rms), kind="linear", fill_value="extrapolate"
        )
        f_onset = interp1d(
            times_orig,
            self._normalize(onset_env),
            kind="linear",
            fill_value="extrapolate",
        )

        N_MELS = 64
        N_FFT = 8192
        MIN_FREQ = 20
        MAX_FREQ = 16000

        S = librosa.feature.melspectrogram(
            y=self.y,
            sr=self.sr,
            n_fft=N_FFT,
            hop_length=hop_length,
            n_mels=N_MELS,
            fmin=MIN_FREQ,
            fmax=MAX_FREQ,
        )

        S_db = librosa.power_to_db(S, ref=np.max)

        min_db = -75.0
        S_norm = (S_db - min_db) / (0 - min_db)
        S_norm = np.clip(S_norm, 0, 1)

        for i in range(N_MELS):
            ratio = i / N_MELS
            if i < 2:
                S_norm[i, :] *= 0.7
            boost = 1.0 + (ratio * 0.5)
            S_norm[i, :] *= boost

        S_norm = S_norm**3.0

        spectrum_frames = []
        for i in range(N_MELS):
            freq_line = S_norm[i, :]
            f_spec = interp1d(
                times_orig, freq_line, kind="linear", fill_value="extrapolate"
            )
            spectrum_frames.append(f_spec(times_target))

        spectrum_final = np.array(spectrum_frames).T

        smoothed_spectrum = np.zeros_like(spectrum_final)
        current_vals = np.zeros(N_MELS)

        attack = 0.5
        decay = 0.25

        for t in range(self.total_frames):
            new_vals = spectrum_final[t]
            is_attack = new_vals > current_vals
            current_vals[is_attack] = (
                current_vals[is_attack] * (1 - attack) + new_vals[is_attack] * attack
            )
            current_vals[~is_attack] = (
                current_vals[~is_attack] * (1 - decay) + new_vals[~is_attack] * decay
            )
            smoothed_spectrum[t] = current_vals

        return {
            "rms": self._ema(f_rms(times_target), 0.1),
            "onset": self._ema(f_onset(times_target), 0.3),
            "spectrum": smoothed_spectrum,
        }

    def _normalize(self, data):
        val_min, val_max = np.min(data), np.max(data)
        if val_max - val_min == 0:
            return np.zeros_like(data)
        return (data - val_min) / (val_max - val_min)

    def _ema(self, data, alpha):
        s = np.zeros_like(data)
        s[0] = data[0]
        for i in range(1, len(data)):
            s[i] = alpha * data[i] + (1 - alpha) * s[i - 1]
        return s
