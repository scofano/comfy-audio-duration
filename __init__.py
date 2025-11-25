import math
import os
import shutil
import subprocess
from typing import Any, Optional, Tuple
import tempfile
import numpy as np  # Added for data manipulation

# ComfyUI temp directory helper
try:
    import folder_paths
except ImportError:
    folder_paths = None

# NOTE: scipy is not included in all ComfyUI installations by default.
# Users must ensure they have 'scipy' installed (e.g., pip install scipy).
try:
    from scipy.io.wavfile import write as write_wav  # Added for saving audio
except ImportError:
    # Fallback to allow the rest of the node to load even if scipy is missing
    # (The saving functionality will fail, but other path/duration checks might pass)
    def write_wav(*args, **kwargs):
        raise ImportError(
            "scipy not found. Please install it (e.g., pip install scipy) "
            "to enable temporary file saving."
        )


class AudioDurationNode:
    """
    ComfyUI custom node that returns the duration of an audio file.
    It can consume EITHER:
      • audio_path (STRING)  — a filesystem path to an audio file, OR
      • audio (AUDIO)        — an upstream audio object from another node.

    Outputs (in order):
      1) seconds_int (INT)
      2) seconds_float (FLOAT)
      3) minutes_int (INT)
      4) minutes_float (FLOAT)
      5) temp_wav_path (STRING) — full path to the temp WAV file ("" if not created)
    """

    @classmethod
    def INPUT_TYPES(cls):
        # audio_path may be left empty; if so and AUDIO is connected, we use the audio input
        return {
            "required": {
                "audio_path": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "seconds_int",
        "seconds_float",
        "minutes_int",
        "minutes_float",
        "temp_wav_path",
    )
    FUNCTION = "run"
    CATEGORY = "audio/utils"

    # ------------------------- helpers -------------------------

    def _extract_samples_sr(self, audio_obj):
        """Return (samples, sr) or (None, None) from many possible AUDIO shapes, including
        nested dicts and batched lists."""
        # 0) None
        if audio_obj is None:
            return None, None

        # 1) If it's a batch/list, try first non-empty item
        if isinstance(audio_obj, (list, tuple)):
            # handle tuple like (samples, sr)
            if len(audio_obj) == 2 and not isinstance(
                audio_obj[0], (list, tuple, dict)
            ):
                samples, sr = audio_obj
                try:
                    sr = float(sr.item()) if hasattr(sr, "item") else float(sr)
                except Exception:
                    pass
                return samples, sr
            # otherwise iterate for first extractable item
            for item in audio_obj:
                s, r = self._extract_samples_sr(item)
                if s is not None and r is not None:
                    return s, r
            return None, None

        # 2) Dict-like
        if isinstance(audio_obj, dict):
            # direct pairs at top-level
            for x_name in ("samples", "waveform", "audio", "pcm", "array", "data"):
                for sr_name in ("sample_rate", "sr"):
                    x = audio_obj.get(x_name)
                    r = audio_obj.get(sr_name)
                    # if 'audio' is a nested dict, recurse
                    if x_name == "audio" and isinstance(x, dict) and r is None:
                        return self._extract_samples_sr(x)
                    if x is not None and r is not None:
                        return x, r

            # some generators return {"tensor": ..., "sample_rate": ...}
            x = audio_obj.get("tensor")
            r = audio_obj.get("sample_rate", audio_obj.get("sr"))
            if x is not None and r is not None:
                return x, r

            # last chance: look for a nested dict that contains samples+sr
            for v in audio_obj.values():
                if isinstance(v, dict):
                    s, r = self._extract_samples_sr(v)
                    if s is not None and r is not None:
                        return s, r
            return None, None

        # 3) Object attributes
        for sr_name in ("sample_rate", "sr"):
            for x_name in ("samples", "waveform", "audio", "pcm", "data"):
                try:
                    x = getattr(audio_obj, x_name)
                    r = getattr(audio_obj, sr_name)
                    if x is not None and r is not None:
                        return x, r
                except Exception:
                    pass

        return None, None

    def _probe_seconds_ffprobe(self, path: str) -> float:
        if shutil.which("ffprobe") is None:
            raise RuntimeError(
                "ffprobe not found on PATH. Please install FFmpeg and ensure ffprobe is available."
            )
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            secs = float(out)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed: {e}")
        except ValueError:
            raise RuntimeError(
                f"ffprobe returned a non-numeric duration for {path!r}: {out!r}"
            )
        if secs < 0:
            raise RuntimeError(f"Invalid (negative) duration returned: {secs}")
        return secs

    def _duration_from_samples(self, audio_obj: Any) -> Optional[float]:
        # Short-circuit if a numeric duration is provided
        if isinstance(audio_obj, dict):
            d = audio_obj.get("duration")
            if isinstance(d, (int, float)) and d > 0:
                return float(d)

        samples, sr = self._extract_samples_sr(audio_obj)
        if samples is None or not sr:
            return None

        # sr might be a tensor/np scalar
        try:
            sr = float(sr.item()) if hasattr(sr, "item") else float(sr)
        except Exception:
            sr = float(sr)

        # Resolve number of frames for common containers (list/tuple/np/torch)
        try:
            import numpy as _np  # optional
        except Exception:
            _np = None

        try:
            import torch as _torch  # optional
        except Exception:
            _torch = None

        n_frames = None
        try:
            if _torch is not None and isinstance(samples, _torch.Tensor):
                samples = samples.detach().cpu().numpy()  # Convert to numpy
                _np = np

            if _np is not None and isinstance(samples, _np.ndarray):
                # Ensure it's a 2D array [N_frames, N_channels] or [N_frames]
                if samples.ndim == 1:
                    n_frames = samples.shape[0]
                elif samples.ndim == 2:
                    # Assume the longer dimension is the number of frames
                    n_frames = max(samples.shape)
            else:
                # Python sequence fallback
                if (
                    hasattr(samples, "__getitem__")
                    and len(samples) > 0
                    and hasattr(samples[0], "__len__")
                ):
                    n_frames = max(len(ch) for ch in samples)
                else:
                    n_frames = len(samples)
        except Exception:
            try:
                n_frames = len(samples)
            except Exception:
                return None

        if not n_frames or sr <= 0:
            return None
        return float(n_frames) / float(sr)

    def _path_from_audio_obj(self, audio_obj: Any) -> Optional[str]:
        # Straight string path
        if isinstance(audio_obj, str) and os.path.isfile(audio_obj):
            return audio_obj

        # Dict-like with common keys
        if isinstance(audio_obj, dict):
            for key in ("path", "filepath", "file", "audio_path"):
                p = audio_obj.get(key)
                if isinstance(p, str) and os.path.isfile(p):
                    return p
            # Some nodes provide a precomputed duration; handled in _duration_from_samples
            # (no path to return here)

        # Objects with attributes
        for attr in ("path", "filepath", "file", "audio_path"):
            try:
                p = getattr(audio_obj, attr)
                if isinstance(p, str) and os.path.isfile(p):
                    return p
            except Exception:
                pass

        return None

    # ------------------------- NEW HELPER FUNCTION -------------------------

    def _save_audio_to_temp(self, samples: Any, sr: float) -> Optional[str]:
        """
        Saves samples/sr to a temporary WAV file in the ComfyUI temp folder.

        Returns:
            Full path to the temp WAV file, or None on failure.
        """
        try:
            if sr <= 0:
                return None

            # Convert to numpy array if it's not already
            if hasattr(samples, "detach") and hasattr(samples, "cpu") and hasattr(
                samples, "numpy"
            ):
                samples = samples.detach().cpu().numpy()
            elif hasattr(samples, "numpy"):
                samples = samples.numpy()
            samples = np.asarray(samples)

            # Convert shape: from [Channels, Frames] or [Frames] to [Frames, Channels]
            if samples.ndim == 2 and samples.shape[0] < samples.shape[1]:
                # Assuming [C, N] is common in torch/some loaders, convert to [N, C]
                samples = samples.T
            elif samples.ndim > 2:
                # Reduce to 2D by flattening extra dimensions or taking the first batch item
                if samples.shape[-1] < 10:
                    samples = samples.reshape(-1, samples.shape[-1])
                else:
                    samples = samples.flatten()

            # Normalize data type and range for WAV saving (e.g., int16)
            max_val = np.abs(samples).max()
            if max_val <= 1.0:
                # Normalize float samples to int16 range if they are floats
                samples = (samples * 32767).astype(np.int16)

            # Use ComfyUI's temp directory structure if available
            if folder_paths is not None:
                try:
                    base_temp = folder_paths.get_temp_directory()
                except Exception:
                    base_temp = tempfile.gettempdir()
            else:
                base_temp = tempfile.gettempdir()

            output_dir = os.path.join(base_temp, "audio_duration_cache")
            os.makedirs(output_dir, exist_ok=True)

            temp_path = os.path.join(
                output_dir, f"temp_{os.getpid()}_{os.urandom(4).hex()}.wav"
            )

            write_wav(temp_path, int(sr), samples)

            return temp_path

        except Exception as e:
            print(f"Error saving audio to temporary file for duration check: {e}")
            return None

    # ------------------------- main -------------------------

    def run(self, audio_path: str, audio: Any = None) -> Tuple[int, float, int, float, str]:
        # Priority: explicit audio_path if provided, otherwise try AUDIO input
        seconds: Optional[float] = None
        path_to_probe: Optional[str] = None
        temp_wav_path: str = ""

        # Case 1: explicit path string
        if isinstance(audio_path, str) and audio_path.strip():
            if os.path.isfile(audio_path.strip()):
                path_to_probe = audio_path.strip()

        # Case 2: AUDIO input
        if not path_to_probe and audio is not None:
            # a) Try extracting a path first
            path_to_probe = self._path_from_audio_obj(audio)

            # b) If we can extract samples/sr, save a temp WAV in ComfyUI temp dir
            samples, sr = self._extract_samples_sr(audio)
            if samples is not None and sr is not None:
                temp = self._save_audio_to_temp(samples, sr)
                if temp:
                    temp_wav_path = temp
                    # If we didn't find another path, use this temp file for probing
                    if path_to_probe is None:
                        path_to_probe = temp

            # c) If no path yet, try duration from samples/dict metadata
            if path_to_probe is None:
                seconds = self._duration_from_samples(audio)

        # If we still don't have seconds, probe by path (original or temp)
        if seconds is None:
            if not path_to_probe:
                raise RuntimeError(
                    "Provide either a valid audio_path or connect an AUDIO input "
                    "that contains a usable path/samples."
                )
            # Use ffprobe on the file path (either original or temp)
            seconds = self._probe_seconds_ffprobe(path_to_probe)

        # Ensure seconds is valid
        if seconds is None or seconds <= 0:
            raise RuntimeError(
                f"Could not determine a valid duration for the audio source: "
                f"{path_to_probe or 'AUDIO input'}"
            )

        # If no temp WAV was created (e.g., only audio_path was used or no samples),
        # keep the output as an empty string to avoid misleading paths.
        if temp_wav_path is None:
            temp_wav_path = ""

        # Package outputs
        return (
            int(seconds),
            float(seconds),
            int(seconds // 60),
            float(seconds) / 60.0,
            temp_wav_path,
        )


# Expose node(s) to ComfyUI
NODE_CLASS_MAPPINGS = {
    "Audio Duration": AudioDurationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Duration": "Audio - Duration",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
