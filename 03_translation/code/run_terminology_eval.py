"""Q2B: translate a manually authored Italian challenge set, then apply the frozen
prototype terminology checker to the SAME baseline translations.

From the KathAI/03_translation project root:
    python code/run_terminology_eval.py --dry-run
    python code/run_terminology_eval.py

Required files:
    data/terminology_challenge.csv
    code/terminology_correct.py  (the previously supplied intervention, unchanged)

Output (never overwrites an existing file):
    data/terminology_results.csv

Model/inference matches the earlier Q2 evaluation:
    Helsinki-NLP/opus-mt-it-en; num_beams=4; max_new_tokens=128; batch size=8.
The reference English is ONLY written to the output for later human review. It is
not provided to the model or terminology checker. No training or fine-tuning occurs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from terminology_correct import correct_translation

REQUIRED = ("id", "domain", "test_type", "italian", "reference_english", "phenomenon")
EXTRA = (
    "model_english", "corrected_english", "correction_applied", "applied_rules",
    "review_flags", "manual_baseline_correct", "manual_corrected_correct", "manual_notes",
)


def load_cases(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [col for col in REQUIRED if col not in columns]
        if missing:
            raise ValueError(f"Missing challenge CSV columns: {missing}")
        rows = list(reader)
    ids = [r["id"].strip() for r in rows]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError("Challenge IDs must be nonempty and unique.")
    if any(not r["italian"].strip() or not r["reference_english"].strip() for r in rows):
        raise ValueError("Every row needs Italian and expected English text.")
    return rows


def translate(rows: list[dict[str, str]], batch_size: int) -> list[str]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    model_name = "Helsinki-NLP/opus-mt-it-en"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()

    outputs: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            inputs = tokenizer(
                [r["italian"] for r in batch],
                return_tensors="pt", padding=True, truncation=True, max_length=512,
            ).to(device)
            generated = model.generate(
                **inputs, num_beams=4, max_new_tokens=128, do_sample=False,
            )
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            print(f"Translated {len(outputs)}/{len(rows)}")
    return outputs


def prepare_results(rows: list[dict[str, str]], translations: list[str]):
    if len(rows) != len(translations):
        raise ValueError("Number of translations does not match number of source sentences.")
    results = []
    for row, baseline in zip(rows, translations):
        corrected, applied, flags = correct_translation(row["italian"], baseline)
        results.append({
            **{key: row.get(key, "") for key in REQUIRED},
            "model_english": baseline,
            "corrected_english": corrected,
            "correction_applied": "yes" if applied else "no",
            "applied_rules": "; ".join(applied),
            "review_flags": "; ".join(flags),
            "manual_baseline_correct": "",
            "manual_corrected_correct": "",
            "manual_notes": "",
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/terminology_challenge.csv")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/terminology_results.csv")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Validate challenge CSV without loading model")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    rows = load_cases(args.input)
    print(f"Validated {len(rows)} unique challenges from {args.input}")
    if args.dry_run:
        print("Dry run passed. No model loaded or file written.")
        return
    if args.output.resolve() == args.input.resolve() or args.output.exists():
        parser.error(f"Output file must be new and distinct from input: {args.output}")
    baseline = translate(rows, args.batch_size)
    results = prepare_results(rows, baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED + EXTRA)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {args.output}")
    print(f"Changed: {sum(r['correction_applied'] == 'yes' for r in results)}; "
          f"flagged: {sum(bool(r['review_flags']) for r in results)}")
    print("Inspect every baseline/corrected pair manually. Reference English was NOT used by the algorithm.")


if __name__ == "__main__":
    main()
