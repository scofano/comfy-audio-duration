# Audio Duration (ComfyUI Custom Node)

Returns the duration of an audio source, which can be provided either as a filesystem path or as raw audio data (samples and sample rate) from an upstream node. Duration calculation primarily uses `ffprobe` (from FFmpeg) for paths, or calculates the duration directly from samples/SR if no path is available.

## Install
1. Ensure FFmpeg is installed and `ffprobe` is on your PATH.
2. Copy this folder into `ComfyUI/custom_nodes/comfy-audio-duration/`.
3. **Install Python dependencies:** This node requires `scipy` to save raw audio data to a temporary file for duration probing.
   ```bash
   pip install -r requirements.txt
   # OR: pip install numpy scipy
````

4.  Restart ComfyUI.

## Node

**Category:** `audio/utils`
**Name:** `Audio Duration`

### Inputs

The node will prioritize a connected `audio` object over the `audio_path` string.

  - `audio_path` (STRING): Absolute or relative path to an audio file. If left empty, the node will use the `audio` input.
  - `audio` (AUDIO, optional): An upstream audio object, which may contain a path, a pre-computed duration, or raw samples and sample rate.

### Outputs

1.  `seconds_int` (INT)
2.  `seconds_float` (FLOAT)
3.  `minutes_int` (INT)
4.  `minutes_float` (FLOAT)

## Troubleshooting

  - **`RuntimeError: Provide either a valid audio_path or connect an AUDIO input...`**:

      * The node could not find a file path in `audio_path` or in the connected `audio` input.
      * The connected `audio` object does not contain a file path and the node failed to extract valid audio samples and sample rate.
      * **Fix:** Ensure the upstream audio node has successfully loaded an audio file, or manually provide a path in the `audio_path` field.

  - **`ImportError: scipy not found. Please install it...`**:

      * You are attempting to use the node with raw audio data (samples/sr) but the necessary `scipy` library is missing.
      * **Fix:** Run `pip install scipy` in your ComfyUI environment.

  - **"ffprobe not found"**: Install FFmpeg and make sure `ffprobe` is available in your shell (on Windows, either add `ffprobe.exe` to PATH or place it next to the ComfyUI executable).

  - **Non-numeric duration**: The file may be corrupt; try remuxing.

  - **IMPORT FAILED / 0.0 seconds** on Windows portable builds usually means Python couldn’t import the package (before your code even runs). Most common causes:

    1)  **Folder name has a hyphen** (e.g., `comfy-audio-duration`). Python packages can’t be imported with hyphens. **Rename the folder** to `AudioDurationNode` or `comfy_audio_duration`.
    2)  **Missing `__init__.py`** or wrong file casing.
    3)  **Syntax error** from copy/paste.