import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pathlib import Path
from analyze_audio import analyze_audio


BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "reference" / "reference.wav"

output_dir = BASE_DIR / "degraded"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "bandpass_500_4000hz.wav"


# ==================================================
# Filter parameters
# ==================================================

low_cutoff = 500
high_cutoff = 4000
filter_order = 6


# ==================================================
# Load reference audio
# ==================================================

y, sr = sf.read(input_path)

if y.ndim > 1:
    y = np.mean(y, axis=1)


# ==================================================
# Design Butterworth bandpass filter
# ==================================================

sos = butter(
    filter_order,
    [low_cutoff, high_cutoff],
    btype="bandpass",
    fs=sr,
    output="sos"
)


# ==================================================
# Apply filter
# ==================================================

y_filtered = sosfilt(
    sos,
    y
)

y_filtered = np.clip(
    y_filtered,
    -1,
    1
)


# ==================================================
# Save degraded audio
# ==================================================

sf.write(
    output_path,
    y_filtered,
    sr
)

print("Bandpass filtering applied.")
print(f"Low cutoff  : {low_cutoff} Hz")
print(f"High cutoff : {high_cutoff} Hz")
print(f"Saved to    : {output_path}")


# ==================================================
# Analyze
# ==================================================

analyze_audio(
    output_path,
    BASE_DIR / "plots" / "bandpass"
)