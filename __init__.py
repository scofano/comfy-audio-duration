import math
import os
import shutil
import subprocess
from typing import Any, Optional, Tuple
import tempfile
import numpy as np

# ComfyUI temp directory helper
try:
    import folder_paths
except ImportError:
    folder_paths = None

# NOTE: scipy is not included in all ComfyUI installations by default.
try:
    from scipy.io.wavfile import write as write_wav
except ImportError:
    def write_wav(*args, **kwargs):
        raise ImportError(
            "scipy not found. Please install it (e.g., pip install scipy) "
            "to enable temporary file saving."
        )


class AudioDurationNode:
    """
    ComfyUI custom node that returns the duration of an audio file.
    
    Inputs:
      • audio_path (STRING) — filesystem path to an audio file, OR
      • audio (AUDIO)       — upstream audio object from another node
    
    Outputs:
      1) seconds_int (INT)
      2) seconds_float (FLOAT)
      3) minutes_int (INT)
      4) minutes_float (FLOAT)
      5) temp_wav_path (STRING) — full path to temp WAV file (if created)
    """

    @classmethod
    def INPUT_TYPES(cls):
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

    # ==================== Helper Methods ====================

    def _to_numpy(self, data):
        """Convert torch tensor or other array-like to numpy array."""
        if data is None:
            return None
        
        # Handle torch tensors
        if hasattr(data, "detach") and hasattr(data, "cpu"):
            return data.detach().cpu().numpy()
        
        # Handle numpy arrays
        if hasattr(data, "numpy"):
            return data.numpy()
        
        # Already numpy or convert
        return np.asarray(data)

    def _extract_audio_info(self, audio_obj):
        """
        Extract (samples, sample_rate, channels) from various AUDIO formats.
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[float], Optional[int]]
        """
        if audio_obj is None:
            return None, None, None

        samples = None
        sr = None
        channels = None

        # Handle dict-like objects (most common in ComfyUI)
        if isinstance(audio_obj, dict):
            # Try to get samples/waveform
            for key in ("waveform", "samples", "audio", "pcm", "data", "tensor"):
                if key in audio_obj:
                    samples = self._to_numpy(audio_obj[key])
                    break
            
            # Try to get sample rate
            for key in ("sample_rate", "sr"):
                if key in audio_obj:
                    sr = audio_obj[key]
                    break
            
            # Try to get channels
            for key in ("channels", "num_channels", "n_channels"):
                if key in audio_obj:
                    channels = audio_obj[key]
                    break

        # Handle list/tuple (e.g., (samples, sr))
        elif isinstance(audio_obj, (list, tuple)):
            if len(audio_obj) >= 2:
                samples = self._to_numpy(audio_obj[0])
                sr = audio_obj[1]
                if len(audio_obj) >= 3:
                    channels = audio_obj[2]

        # Handle object with attributes
        else:
            for attr in ("waveform", "samples", "audio", "data"):
                if hasattr(audio_obj, attr):
                    samples = self._to_numpy(getattr(audio_obj, attr))
                    break
            
            for attr in ("sample_rate", "sr"):
                if hasattr(audio_obj, attr):
                    sr = getattr(audio_obj, attr)
                    break
            
            for attr in ("channels", "num_channels", "n_channels"):
                if hasattr(audio_obj, attr):
                    channels = getattr(audio_obj, attr)
                    break

        # Convert sr to float
        if sr is not None:
            sr = float(sr.item()) if hasattr(sr, "item") else float(sr)

        # Convert channels to int
        if channels is not None:
            channels = int(channels.item()) if hasattr(channels, "item") else int(channels)

        return samples, sr, channels

    def _infer_channels_from_shape(self, samples):
        """
        Infer number of channels from array shape.
        
        Returns:
            int or None
        """
        if samples is None or not isinstance(samples, np.ndarray):
            return None
        
        if samples.ndim == 1:
            # Cannot reliably infer from 1D array
            return None
        elif samples.ndim == 2:
            # Smaller dimension is likely channels
            # Common formats: [channels, frames] or [frames, channels]
            return min(samples.shape)
        else:
            # Higher dimensions - take the smallest
            return min(samples.shape)

    def _calculate_duration_from_samples(self, samples, sr, channels):
        """
        Calculate duration in seconds from audio samples.
        
        Args:
            samples: numpy array of audio data
            sr: sample rate (Hz)
            channels: number of channels (or None)
        
        Returns:
            float: duration in seconds, or None if cannot calculate
        """
        if samples is None or sr is None or sr <= 0:
            return None

        samples = self._to_numpy(samples)
        if samples is None:
            return None

        # Infer channels from shape if not provided
        if channels is None:
            channels = self._infer_channels_from_shape(samples)

        # Calculate number of frames based on array shape
        if samples.ndim == 1:
            # 1D array: could be mono OR interleaved stereo/multi-channel
            total_samples = len(samples)
            if channels is not None and channels > 1:
                # Interleaved format: [L1, R1, L2, R2, ...]
                n_frames = total_samples // channels
            else:
                # Assume mono if channels unknown
                n_frames = total_samples
                if channels is None:
                    print(
                        f"Warning: 1D audio array without channel info. "
                        f"Assuming mono ({total_samples} samples). "
                        f"If this is stereo, duration will be 2x actual."
                    )
        
        elif samples.ndim == 2:
            # 2D array: [channels, frames] or [frames, channels]
            # The larger dimension is more likely frames
            if samples.shape[0] > samples.shape[1]:
                n_frames = samples.shape[0]  # [frames, channels]
            else:
                n_frames = samples.shape[1]  # [channels, frames]
        
        else:
            # 3D+ arrays: flatten or take max dimension
            n_frames = max(samples.shape)

        if n_frames <= 0:
            return None

        return float(n_frames) / float(sr)

    def _extract_file_path(self, audio_obj):
        """Extract file path from audio object if available."""
        if isinstance(audio_obj, str) and os.path.isfile(audio_obj):
            return audio_obj

        if isinstance(audio_obj, dict):
            for key in ("path", "filepath", "file", "audio_path"):
                p = audio_obj.get(key)
                if isinstance(p, str) and os.path.isfile(p):
                    return p

        for attr in ("path", "filepath", "file", "audio_path"):
            try:
                p = getattr(audio_obj, attr)
                if isinstance(p, str) and os.path.isfile(p):
                    return p
            except Exception:
                pass

        return None

    def _probe_duration_ffprobe(self, path: str) -> float:
        """Use ffprobe to get duration of audio file."""
        if shutil.which("ffprobe") is None:
            raise RuntimeError(
                "ffprobe not found. Please install FFmpeg."
            )
        
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            duration = float(out)
        except (subprocess.CalledProcessError, ValueError) as e:
            raise RuntimeError(f"ffprobe failed for {path}: {e}")
        
        if duration < 0:
            raise RuntimeError(f"Invalid duration: {duration}")
        
        return duration

    def _save_temp_wav(self, samples, sr, channels=None):
        """
        Save audio samples to a temporary WAV file.
        
        Returns:
            str: path to temp file, or None on failure
        """
        try:
            if sr <= 0:
                return None

            samples = self._to_numpy(samples)
            if samples is None:
                return None

            # Infer channels if not provided
            if channels is None:
                channels = self._infer_channels_from_shape(samples)

            # Reshape to [frames, channels] for WAV writing
            if samples.ndim == 1:
                if channels is not None and channels > 1:
                    # Reshape interleaved to [frames, channels]
                    n_frames = len(samples) // channels
                    samples = samples[:n_frames * channels].reshape(n_frames, channels)
                else:
                    # Keep as 1D for mono
                    samples = samples.reshape(-1, 1)
            
            elif samples.ndim == 2:
                # Ensure [frames, channels] format
                if samples.shape[0] < samples.shape[1]:
                    # Likely [channels, frames], transpose
                    samples = samples.T

            # Normalize to int16 range if needed
            if samples.dtype.kind == 'f':  # float type
                max_val = np.abs(samples).max()
                if max_val > 0:
                    samples = (samples / max_val * 32767).astype(np.int16)
                else:
                    samples = samples.astype(np.int16)
            elif samples.dtype != np.int16:
                samples = samples.astype(np.int16)

            # Determine temp directory
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
                output_dir,
                f"temp_{os.getpid()}_{os.urandom(4).hex()}.wav"
            )

            write_wav(temp_path, int(sr), samples)
            return temp_path

        except Exception as e:
            print(f"Error saving temp WAV: {e}")
            return None

    # ==================== Main Function ====================

    def run(
        self,
        audio_path: str,
        audio: Any = None
    ) -> Tuple[int, float, int, float, str]:
        """
        Main execution function.
        
        Returns:
            (seconds_int, seconds_float, minutes_int, minutes_float, temp_wav_path)
        """
        duration = None
        file_path = None
        temp_wav_path = ""

        # Priority 1: Explicit audio_path string
        if isinstance(audio_path, str) and audio_path.strip():
            path = audio_path.strip()
            if os.path.isfile(path):
                file_path = path

        # Priority 2: AUDIO object
        if audio is not None:
            # Try to extract existing file path
            if file_path is None:
                file_path = self._extract_file_path(audio)

            # Try to calculate from samples
            samples, sr, channels = self._extract_audio_info(audio)
            
            if samples is not None and sr is not None:
                # Calculate duration directly
                duration = self._calculate_duration_from_samples(samples, sr, channels)
                
                # Also save a temp WAV for reference/debugging
                temp = self._save_temp_wav(samples, sr, channels)
                if temp:
                    temp_wav_path = temp
                    # Use temp file for probing if we don't have another path
                    if file_path is None:
                        file_path = temp

        # Fallback: probe file with ffprobe if we haven't calculated duration yet
        if duration is None:
            if file_path is None:
                raise RuntimeError(
                    "No valid audio source provided. "
                    "Please provide either audio_path or connect an AUDIO input."
                )
            duration = self._probe_duration_ffprobe(file_path)

        # Validate
        if duration is None or duration <= 0:
            raise RuntimeError(f"Invalid duration calculated: {duration}")

        # Return outputs
        return (
            int(duration),                  # seconds_int
            float(duration),                # seconds_float
            int(duration // 60),            # minutes_int
            float(duration / 60.0),         # minutes_float
            temp_wav_path,                  # temp_wav_path
        )


# ==================== ComfyUI Registration ====================

NODE_CLASS_MAPPINGS = {
    "Audio Duration": AudioDurationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Duration": "Audio - Duration",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
