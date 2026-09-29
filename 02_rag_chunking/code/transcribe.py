import os
import shutil
from pathlib import Path

import whisper


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "transcripts" / "raw"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FFmpeg
# --------------------------------------------------

FFMPEG_BIN = Path(
    r"C:\ffmpeg-n9.0-latest-win64-gpl-9.0"
    r"\ffmpeg-n9.0-latest-win64-gpl-9.0"
    r"\bin"
)

if not (FFMPEG_BIN / "ffmpeg.exe").exists():
    raise FileNotFoundError(
        f"ffmpeg.exe was not found in:\n{FFMPEG_BIN}"
    )

# Make FFmpeg visible to Whisper/Python
os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ["PATH"]

# Verify that Python can now find FFmpeg
ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    raise RuntimeError("Python still cannot locate FFmpeg.")

print(f"FFmpeg found: {ffmpeg_path}")


# --------------------------------------------------
# Whisper
# --------------------------------------------------

print("Loading Whisper model...")

model = whisper.load_model("small")


# --------------------------------------------------
# Find audio
# --------------------------------------------------

audio_files = list(VIDEOS_DIR.glob("*.mp3"))

print(f"Found {len(audio_files)} audio file(s).\n")

if not audio_files:
    raise FileNotFoundError(
        f"No MP3 files found in:\n{VIDEOS_DIR}"
    )


# --------------------------------------------------
# Transcribe
# --------------------------------------------------

for audio in audio_files:

    print(f"Transcribing {audio.name}...")

    result = model.transcribe(
        str(audio),
        language="en",
        verbose=False
    )

    output_file = OUTPUT_DIR / f"{audio.stem}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Saved -> {output_file}\n")


print("Done.")