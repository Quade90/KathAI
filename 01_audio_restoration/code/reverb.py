import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from pathlib import Path
from analyze_audio import analyze_audio


BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "reference" / "reference.wav"

output_dir = BASE_DIR / "degraded"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "reverb.wav"


# ==================================================
# Load reference audio
# ==================================================

y, sr = sf.read(input_path)

if y.ndim > 1:
    y = np.mean(y, axis=1)


# ==================================================
# Create synthetic room impulse response
# ==================================================

duration = 0.8

h = np.zeros(
    int(duration * sr)
)


# ==================================================
# Direct sound
# ==================================================

h[0] = 1.0


# ==================================================
# Early reflections
# ==================================================

reflections = [
    (0.020, 0.50),
    (0.045, 0.30),
    (0.080, 0.20),
    (0.120, 0.10),
]

for delay, amplitude in reflections:

    index = int(delay * sr)

    h[index] = amplitude


# ==================================================
# Reverberation tail
# ==================================================

tail_start = int(0.15 * sr)
tail_end = len(h)

t = np.arange(
    tail_end - tail_start
) / sr

decay_rate = 5.0

rng = np.random.default_rng(42)

noise = rng.normal(
    0,
    1,
    len(t)
)

h[tail_start:] += (
    0.08
    * noise
    * np.exp(-decay_rate * t)
)


# ==================================================
# Apply reverb
# ==================================================

y_reverb = fftconvolve(
    y,
    h,
    mode="full"
)


# ==================================================
# Normalize
# ==================================================

y_reverb = (
    y_reverb
    / np.max(np.abs(y_reverb))
)

y_reverb *= 0.95


# ==================================================
# Save degraded audio
# ==================================================

sf.write(
    output_path,
    y_reverb,
    sr
)

print("Synthetic reverberation applied.")
print(f"Room response duration : {duration} s")
print(f"Saved to               : {output_path}")


# ==================================================
# Analyze
# ==================================================

analyze_audio(
    output_path,
    BASE_DIR / "plots" / "reverb"
)