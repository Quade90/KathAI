
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "flores_dev.csv"
OUTPUT_FILE = ROOT / "data" / "eval_sample.csv"

SAMPLE_SIZE = 50
SEED = 42

random.seed(SEED)

# Load development examples only
with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# Sort by source sentence length
rows.sort(key=lambda r: len(r["italian"].split()))

# Divide into five length-based groups
groups = [
    rows[i * len(rows) // 5 : (i + 1) * len(rows) // 5]
    for i in range(5)
]

# Sample 10 sentences from each group
selected = []

for group in groups:
    selected.extend(random.sample(group, SAMPLE_SIZE // 5))

# Restore original dataset order
selected.sort(key=lambda r: int(r["id"]))

# Export
with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "italian", "english"]
    )

    writer.writeheader()
    writer.writerows(selected)

print(f"Selected {len(selected)} examples.")
print(f"Saved to: {OUTPUT_FILE}")
