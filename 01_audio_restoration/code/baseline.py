from pathlib import Path
from analyze_audio import analyze_audio


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "reference" / "reference.wav"

output_dir = BASE_DIR / "plots" / "reference"


# ==================================================
# Analyze reference audio
# ==================================================

analyze_audio(
    input_path,
    output_dir
)