from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "perturbation_challenge.csv"
OUTPUT_FILE = ROOT / "data" / "perturbation_results.csv"

MODEL_NAME = "Helsinki-NLP/opus-mt-it-en"
BATCH_SIZE = 8


def translate(texts, tokenizer, model, device):
    outputs = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                num_beams=4,
                max_new_tokens=128
            )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        outputs.extend(decoded)

    return outputs


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. "
            "Delete or move it before rerunning."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"id", "type", "original_it", "perturbed_it"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Input must contain columns: {sorted(required)}"
        )

    if df["id"].duplicated().any():
        raise ValueError("Challenge IDs must be unique.")

    print(f"Loaded {len(df)} perturbation pairs.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading {MODEL_NAME} on {device}...")

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Translate originals and perturbations independently.
    original_outputs = translate(
        df["original_it"].tolist(),
        tokenizer,
        model,
        device
    )

    perturbed_outputs = translate(
        df["perturbed_it"].tolist(),
        tokenizer,
        model,
        device
    )

    results = df.copy()

    results["original_en"] = original_outputs
    results["perturbed_en"] = perturbed_outputs

    # Leave these for manual semantic inspection.
    results["semantic_effect"] = ""
    results["notes"] = ""

    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved results to: {OUTPUT_FILE}")
    print()
    print(
        "Manually fill semantic_effect with "
        "'stable', 'minor_change', or 'significant_change'."
    )


if __name__ == "__main__":
    main()