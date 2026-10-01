import numpy as np
import soundfile as sf
from pathlib import Path
from analyze_audio import analyze_audio


BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "reference" / "reference.wav"

output_dir = BASE_DIR / "degraded"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "clipped_03.wav"

# ==================================================
# Clipping parameter
# ==================================================

threshold = 0.3


# ==================================================
# Load reference audio
# ==================================================

y, sr = sf.read(input_path)

if y.ndim > 1:
    y = np.mean(y, axis=1)


# ==================================================
# Apply hard clipping
# ==================================================

y_clipped = np.clip(
    y,
    -threshold,
    threshold
)


# ==================================================
# Save degraded audio
# ==================================================

sf.write(
    output_path,
    y_clipped,
    sr
)

print("Clipping applied.")
print(f"Threshold : ±{threshold}")
print(f"Saved to  : {output_path}")


# ==================================================
# Analyze
# ==================================================

analyze_audio(
    output_path,
    BASE_DIR / "plots" / "clipping"
)