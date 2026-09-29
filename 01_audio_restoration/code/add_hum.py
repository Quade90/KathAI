import numpy as np
import soundfile as sf
from pathlib import Path
from analyze_audio import analyze_audio


BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "reference" / "reference.wav"

output_dir = BASE_DIR / "degraded"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "hum_50hz.wav"


# ==================================================
# Hum parameters
# ==================================================

frequency = 50
amplitude = 0.3


# ==================================================
# Load reference audio
# ==================================================

y, sr = sf.read(input_path)

if y.ndim > 1:
    y = np.mean(y, axis=1)


# ==================================================
# Generate 50 Hz hum
# ==================================================

t = np.arange(len(y)) / sr

hum = amplitude * np.sin(
    2 * np.pi * frequency * t
)


# ==================================================
# Add hum
# ==================================================

y_hum = y + hum

y_hum = np.clip(
    y_hum,
    -1,
    1
)


# ==================================================
# Save degraded audio
# ==================================================

sf.write(
    output_path,
    y_hum,
    sr
)

print("50 Hz hum added.")
print(f"Frequency : {frequency} Hz")
print(f"Amplitude : {amplitude}")
print(f"Saved to  : {output_path}")


# ==================================================
# Analyze
# ==================================================

analyze_audio(
    output_path,
    BASE_DIR / "plots" / "hum"
)