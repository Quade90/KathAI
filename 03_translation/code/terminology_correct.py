"""Conservative, source-aware Italian -> English terminology correction.

Development-stage rules, derived from manually reviewed FLORES+ dev examples.
MarianMT itself is NOT retrained or modified. Do not tune this on devtest.

Run from 03_translation/:
    python code/terminology_correct.py
    python code/terminology_correct.py --self-test
    python code/terminology_correct.py --input data/baseline_results.csv --output data/development_review.csv

Input must have columns 'italian' and 'model_english'. All original columns
are retained; corrected text and the audit trail are appended.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    name: str
    source: str             # Italian pattern that licenses the correction
    wrong_english: str      # Only change this specific MarianMT error
    correct_english: str
    context: str | None = None  # Additional Italian evidence, if necessary


# These rules are DEVELOPMENT hypotheses, not a general Italian dictionary.
# Match the SOURCE and the actual mistranslation before changing the target.
# More specific multi-word expressions are safer than isolated word swaps.
RULES = [
    Rule(
        name="aerospace_command_module",
        source=r"\bmodulo\s+di\s+comando\b",
        wrong_english=r"\bcontrol module\b",
        correct_english="command module",
    ),
    Rule(
        name="geology_lobate_scarps",
        source=r"\bscarpate\s+lobate\b",
        wrong_english=r"\blobed shoes\b",
        correct_english="lobate scarps",
    ),
    Rule(
        name="fusion_distinction",
        source=r"\bsi\s+uniscono\s*\(\s*o\s+si\s+fondono\s*\)",
        wrong_english=r"\bmerge\s*\(\s*or\s+merge\s*\)",
        correct_english="join (or fuse)",
    ),
    Rule(
        name="hospitality_host",
        source=r"\bpadrone\s+di\s+casa\b",
        context=r"\b(colazione|ospit\w*|alloggi\w*|pension\w*|bed\s*(?:and|&)\s*breakfast)\b",
        wrong_english=r"\blandlord\b",
        correct_english="host",
    ),
    Rule(
        name="gps_maps",
        source=r"\bcarte\s+per\s+(?:il\s+)?GPS\b",
        wrong_english=r"\bcards\s+for\s+(?:the\s+)?GPS\b",
        correct_english="maps for GPS",
    ),
    Rule(
        name="standalone_gps",
        source=r"\bGPS\s+autonomo\b",
        wrong_english=r"\ban\s+autonomous\s+GPS\s+device\b",
        correct_english="a standalone GPS device",
    ),
]

# The source for ID 500 contains 'piegatori', but does not itself explicitly
# say paper/origami. Consequently, the system should FLAG, not blindly infer.
AMBIGUITY_CHECKS = [
    (
        "piegatori_requires_activity_context",
        r"\bpiegator[ie]\b",
        r"\bbenders?\b",
        r"\b(cart[ae]|origami|piegatur\w*|fogli\w*)\b",
    ),
]


def same_case(replacement: str, matched: str) -> str:
    """Preserve capitalization of the first word where possible."""
    if matched.isupper():
        return replacement.upper()
    if matched[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def correct_translation(italian: str, english: str) -> tuple[str, list[str], list[str]]:
    corrected = english
    applied: list[str] = []
    flags: list[str] = []

    for rule in RULES:
        if not re.search(rule.source, italian, flags=re.IGNORECASE):
            continue
        if rule.context and not re.search(rule.context, italian, flags=re.IGNORECASE):
            flags.append(f"{rule.name}: source context not confirmed")
            continue
        if not re.search(rule.wrong_english, corrected, flags=re.IGNORECASE):
            # Not necessarily a failure: MarianMT may already have it right,
            # or have made a different error outside this limited rule set.
            continue
        corrected, n = re.subn(
            rule.wrong_english,
            lambda match: same_case(rule.correct_english, match.group()),
            corrected,
            count=1,
            flags=re.IGNORECASE,
        )
        if n:
            applied.append(rule.name)

    for name, source, target, disambiguating_context in AMBIGUITY_CHECKS:
        if (re.search(source, italian, flags=re.IGNORECASE)
                and re.search(target, corrected, flags=re.IGNORECASE)):
            if not re.search(disambiguating_context, italian, flags=re.IGNORECASE):
                flags.append(f"{name}: insufficient source context; manual review")
            else:
                # Even with a contextual keyword, this prototype does not
                # automatically rewrite an isolated occupation/activity term.
                flags.append(f"{name}: inspect folding versus bending meaning")

    return corrected, applied, flags


def process_file(input_path: Path, output_path: Path) -> dict[str, int]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must not overwrite the baseline input.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing file: {output_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original_columns = reader.fieldnames or []
        required = {"italian", "model_english"}
        if not required.issubset(original_columns):
            raise ValueError(f"Missing required columns: {sorted(required - set(original_columns))}")
        rows = list(reader)

    extra = ["corrected_english", "correction_applied", "applied_rules", "review_flags"]
    if any(name in original_columns for name in extra):
        raise ValueError("Input already contains intervention columns. Use the baseline file.")

    summary = {"rows": len(rows), "changed": 0, "flagged": 0, "rule_firings": 0}
    for row in rows:
        corrected, applied, flags = correct_translation(row["italian"], row["model_english"])
        row["corrected_english"] = corrected
        row["correction_applied"] = "yes" if applied else "no"
        row["applied_rules"] = "; ".join(applied)
        row["review_flags"] = "; ".join(flags)
        summary["changed"] += bool(applied)
        summary["flagged"] += bool(flags)
        summary["rule_firings"] += len(applied)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=original_columns + extra)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def self_test() -> None:
    cases = [
        ("ritorno del modulo di comando Apollo 10", "return of the Apollo 10 control module",
         "return of the Apollo 10 command module", ["aerospace_command_module"]),
        ("Le scarpate lobate sulla Luna", "The lobed shoes on the Moon",
         "The lobate scarps on the Moon", ["geology_lobate_scarps"]),
        ("gli atomi di idrogeno si uniscono (o si fondono) tra loro",
         "hydrogen atoms merge (or merge) together", "hydrogen atoms join (or fuse) together",
         ["fusion_distinction"]),
        ("La colazione comprende una specialità del padrone di casa.",
         "Breakfast includes a specialty of the landlord.",
         "Breakfast includes a specialty of the host.", ["hospitality_host"]),
        ("Il padrone di casa deve riparare il tetto.", "The landlord must repair the roof.",
         "The landlord must repair the roof.", []),
        ("I metalli si fondono ad alta temperatura.", "Metals merge at high temperatures.",
         "Metals merge at high temperatures.", []),
        ("i piegatori inesperti", "inexperienced benders", "inexperienced benders", []),
        ("nuove carte per il GPS o un dispositivo GPS autonomo",
         "new cards for GPS or an autonomous GPS device",
         "new maps for GPS or a standalone GPS device", ["gps_maps", "standalone_gps"]),
    ]
    for index, (src, baseline, expected, expected_rules) in enumerate(cases, start=1):
        output, rules, flags = correct_translation(src, baseline)
        assert output == expected, (index, output, expected)
        assert rules == expected_rules, (index, rules, expected_rules)
        if index == 7:
            assert flags, "Ambiguous piegatori must be flagged."
    print(f"PASS: {len(cases)} positive/negative tests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/baseline_results.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/development_review.csv")
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = process_file(args.input, args.output)
    print(f"Input: {args.input}\nOutput: {args.output}")
    print(", ".join(f"{name}={value}" for name, value in result.items()))
    print("Review changed rows and flags manually before claiming an improvement.")


if __name__ == "__main__":
    main()
