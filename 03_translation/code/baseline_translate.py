"""Part 2.3: unmodified Italian -> English translation baseline (smoke test).

From the 03_translation/ directory:
    python code/baseline_translate.py

Dependencies:
    python -m pip install -U "transformers>=4.44,<5" torch sentencepiece sacremoses

The first run downloads the model/tokenizer from Hugging Face and caches them.
This script only reads the FLORES+ dev split; devtest stays untouched.
"""

import csv
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "Helsinki-NLP/opus-mt-it-en"
ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "flores_dev.csv"
OUTPUT_CSV = ROOT / "data" / "baseline_preview.csv"
N_EXAMPLES = 5


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing {INPUT_CSV}; run download_flores.py first.")

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        examples = [row for _, row in zip(range(N_EXAMPLES), reader)]

    if not examples or any(not row.get("italian") or not row.get("english") for row in examples):
        raise ValueError("No valid aligned Italian-English examples found.")

    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Device: {device}")

    sources = [row["italian"] for row in examples]
    batch = tokenizer(sources, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**batch, num_beams=4, max_new_tokens=128)
    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "italian", "reference_english", "model_english"])
        for row, prediction in zip(examples, predictions):
            writer.writerow([row["id"], row["italian"], row["english"], prediction])
            print(f"\nID: {row['id']}")
            print(f"IT:    {row['italian']}")
            print(f"REF:   {row['english']}")
            print(f"MODEL: {prediction}")

    print(f"\nSaved {len(predictions)} translations to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
