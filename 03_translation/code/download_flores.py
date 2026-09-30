"""Download aligned Italian–English FLORES+ evaluation data.

Prerequisites:
    pip install -U datasets huggingface_hub
    hf auth login
Also accept the gated dataset conditions at:
https://huggingface.co/datasets/openlanguagedata/flores_plus

Run from the 03_translation directory:
    python code/download_flores.py
"""
from pathlib import Path
import csv

from datasets import load_dataset

REPO = "openlanguagedata/flores_plus"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Italian FLORES+ data...")
italian = load_dataset(REPO, "ita_Latn")
print("Loading English FLORES+ data...")
english = load_dataset(REPO, "eng_Latn")

expected_counts = {"dev": 997, "devtest": 1012}
for split in ("dev", "devtest"):
    en_by_id = {str(row["id"]): row["text"] for row in english[split]}
    it_by_id = {str(row["id"]): row["text"] for row in italian[split]}

    if len(en_by_id) != len(english[split]) or len(it_by_id) != len(italian[split]):
        raise ValueError(f"Duplicate sentence IDs detected in {split}.")
    if en_by_id.keys() != it_by_id.keys():
        raise ValueError(f"English/Italian sentence IDs do not match in {split}.")
    if len(it_by_id) != expected_counts[split]:
        raise ValueError(
            f"Unexpected {split} count: {len(it_by_id)} "
            f"(expected {expected_counts[split]})."
        )

    destination = DATA_DIR / f"flores_{split}.csv"
    with destination.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "italian", "english"])
        for sentence_id in sorted(it_by_id, key=int):
            writer.writerow([sentence_id, it_by_id[sentence_id], en_by_id[sentence_id]])

    print(f"Saved {len(it_by_id)} aligned pairs to {destination}")

print("Dataset download and alignment complete.")
