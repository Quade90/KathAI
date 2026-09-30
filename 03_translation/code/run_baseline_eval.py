
"""Generate MarianMT translations for the 50 selected FLORES+ dev examples.

Run from the 03_translation directory:
    python code/run_baseline_eval.py

This script does not access the held-out devtest split.
"""

import csv
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_ID = "Helsinki-NLP/opus-mt-it-en"

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "eval_sample.csv"
OUTPUT_FILE = ROOT / "data" / "baseline_results.csv"

BATCH_SIZE = 8


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    # Prevent accidentally overwriting manual annotations.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists.\n"
            "Rename or move it before rerunning this script "
            "to avoid losing your manual suggestions."
        )

    with INPUT_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} evaluation examples.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print("Loading MarianMT...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    model.to(device)
    model.eval()

    results = []

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        sentences = [row["italian"] for row in batch]

        inputs = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=128,
            )

        translations = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        for row, translation in zip(batch, translations):
            results.append({
                "id": row["id"],
                "italian": row["italian"],
                "reference_english": row["english"],
                "model_english": translation,
                "my_suggestions": "",
                "semantic_preserved": "",
                "error_category": "",
                "review_notes": "",
            })

        print(
            f"Translated {len(results)}/{len(rows)} sentences."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "italian",
        "reference_english",
        "model_english",
        "my_suggestions",
        "semantic_preserved",
        "error_category",
        "review_notes",
    ]

    with OUTPUT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nCompleted. Saved to: {OUTPUT_FILE}")
    print(
        "Fill the 'my_suggestions' column during manual review. "
        "Leave the remaining review columns blank."
    )


if __name__ == "__main__":
    main()
