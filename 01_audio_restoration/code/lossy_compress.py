import subprocess
from pathlib import Path
from analyze_audio import analyze_audio


BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "reference" / "reference.wav"

output_dir = BASE_DIR / "degraded"
output_dir.mkdir(exist_ok=True)

mp3_path = output_dir / "compressed_32kbps.mp3"
output_path = output_dir / "compressed_32kbps.wav"


# ==================================================
# FFmpeg location
# ==================================================

ffmpeg = Path(
    r"C:\ffmpeg-n9.0-latest-win64-gpl-9.0"
    r"\ffmpeg-n9.0-latest-win64-gpl-9.0"
    r"\bin\ffmpeg.exe"
)


# ==================================================
# Compress to 32 kbps MP3
# ==================================================

subprocess.run(
    [
        str(ffmpeg),
        "-y",
        "-i",
        str(input_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "32k",
        str(mp3_path)
    ],
    check=True
)


# ==================================================
# Decode MP3 back to WAV
# ==================================================

subprocess.run(
    [
        str(ffmpeg),
        "-y",
        "-i",
        str(mp3_path),
        "-acodec",
        "pcm_s16le",
        str(output_path)
    ],
    check=True
)


print("Low-bitrate lossy compression applied.")
print("Bitrate : 32 kbps")
print(f"MP3     : {mp3_path}")
print(f"WAV     : {output_path}")


# ==================================================
# Analyze
# ==================================================

analyze_audio(
    output_path,
    BASE_DIR / "plots" / "lossy_compress"
)