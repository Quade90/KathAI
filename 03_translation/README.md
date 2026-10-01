# 2.3 — When the English Sounds Right but the Meaning Is Wrong

This section investigates semantic reliability in Italian-to-English machine translation.

The experiments use MarianMT as the baseline translator, analyze fluent-but-incorrect outputs, test sensitivity to small source perturbations, and implement a conservative terminology-aware correction layer.

## Model and Dataset

Translation model:

```text
Helsinki-NLP/opus-mt-it-en
```

Dataset:

```text
FLORES+
Italian: ita_Latn
English: eng_Latn
```

The development split is used for the main investigation and terminology-rule development.

The FLORES+ `devtest` split is downloaded but left untouched by the evaluation scripts.

## Code

```text
code/
├── download_flores.py
├── baseline_translate.py
├── build_eval_sample.py
├── run_baseline_eval.py
├── terminology_correct.py
├── run_terminology_eval.py
└── test_perturbations.py
```

The repository keeps the small, hand-authored challenge sets required to define the experiments:

```text
data/
├── terminology_challenge.csv
└── perturbation_challenge.csv
```

Downloaded FLORES+ files and generated result CSVs are intentionally not committed.

## Setup

Install the repository requirements from the project root:

```bash
pip install -r requirements.txt
```

To download FLORES+, accept the dataset conditions on Hugging Face and authenticate if required.

## Download and Align FLORES+

`download_flores.py`

The script downloads Italian and English independently and aligns them by sentence ID.

It verifies:

```text
dev:     997 aligned sentence pairs
devtest: 1012 aligned sentence pairs
```

Run:

```bash
python code/download_flores.py
```

Generated files:

```text
data/flores_dev.csv
data/flores_devtest.csv
```

These are ignored by Git because they can be regenerated.

## Baseline Smoke Test

`baseline_translate.py`

This translates the first five aligned development examples using:

```text
num_beams = 4
max_new_tokens = 128
```

Run:

```bash
python code/baseline_translate.py
```

Generated output:

```text
data/baseline_preview.csv
```

## Evaluation Sample

`build_eval_sample.py`

A 50-sentence development sample is constructed to avoid evaluating only one sentence-length regime.

Procedure:

1. sort FLORES+ `dev` by Italian sentence length
2. split it into five length-based groups
3. randomly sample 10 sentences from each group
4. use `SEED = 42`
5. restore the selected examples to original dataset order

Run:

```bash
python code/build_eval_sample.py
```

Generated output:

```text
data/eval_sample.csv
```

## Baseline Translation Evaluation

`run_baseline_eval.py`

The 50 selected examples are translated with the same MarianMT configuration:

```text
batch size = 8
beam size = 4
max_new_tokens = 128
```

The output includes columns for later manual review.

Run:

```bash
python code/run_baseline_eval.py
```

Generated output:

```text
data/baseline_results.csv
```

The script refuses to overwrite an existing result file to avoid destroying manual annotations.

## Observed Error Types

Manual inspection identified several failure categories:

- domain/context-specific terminology
- wrong word-sense selection
- loss of specificity / ambiguity
- referent / pronoun errors
- relationship / argument errors
- named-entity / proper-name errors
- awkward English without major semantic loss

Representative examples included:

```text
modulo di comando
→ "control module"
instead of "command module"
```

```text
scarpate lobate
→ "lobed shoes"
instead of "lobate scarps"
```

```text
prevent enemy troops from escaping
→ "prevent the escape from enemy troops"
```

The repeated failure selected for intervention was **domain/context-sensitive terminology**.

## Conservative Terminology Intervention

`terminology_correct.py`

MarianMT itself is not retrained or fine-tuned.

Instead, a small post-translation correction layer is applied after the baseline translation.

A correction is allowed only when:

1. the relevant Italian source pattern is present
2. any required source context is present
3. the specific known MarianMT mistranslation is present in the English output

The layer therefore does not perform blind global word replacement.

### Development Rules

The prototype contains rules for cases including:

- `modulo di comando`: `control module` → `command module`
- `scarpate lobate`: `lobed shoes` → `lobate scarps`
- `si uniscono (o si fondono)`: `merge (or merge)` → `join (or fuse)`
- hospitality context: `landlord` → `host`
- `carte per il GPS`: `cards for GPS` → `maps for GPS`
- `GPS autonomo`: `an autonomous GPS device` → `a standalone GPS device`

An ambiguous translation of `piegatori` as `benders` is flagged for manual review instead of being automatically replaced.

## Intervention Self-Test

Run:

```bash
python code/terminology_correct.py --self-test
```

Expected result:

```text
PASS: 8 positive/negative tests
```

## Development-Set Intervention

Apply the correction layer to the 50 baseline translations:

```bash
python code/terminology_correct.py
```

Generated output:

```text
data/development_review.csv
```

Observed summary:

```text
rows = 50
changed = 5
flagged = 1
rule_firings = 6
```

These rules are treated as development hypotheses and are frozen before the separate terminology challenge evaluation.

## Held-Out Terminology Challenge

`run_terminology_eval.py`

A separate 20-sentence controlled challenge set is used to test whether the correction layer generalizes beyond the examples used while developing the rules.

The set contains:

```text
20 total cases
14 target cases
6 control/non-target cases
```

The English reference is stored only for later human evaluation.

It is **not** provided to:

- MarianMT
- the correction layer

No training or fine-tuning occurs.

### Validate the Challenge Set

```bash
python code/run_terminology_eval.py --dry-run
```

### Run the Evaluation

```bash
python code/run_terminology_eval.py
```

Generated output:

```text
data/terminology_results.csv
```

### Results

| Result | Count |
|---|---:|
| Challenge sentences | 20 |
| Automatically changed | 3 |
| Successfully corrected | 3 |
| Flagged for human review | 1 |
| Observed regressions | 0 |

Successful automatic corrections included:

| Case | Baseline | Corrected |
|---|---|---|
| A1 | control module | command module |
| E3 | landlord | host |
| E5 | autonomous GPS device | standalone GPS device |

The successful cases were lexically and structurally close enough to the development cases for both the source-side and mistranslation patterns to match.

## Important Misses

The intervention is intentionally conservative, which limits coverage.

### G1

```text
Baseline: lobed escarpments
Expected terminology: lobate scarps
```

The development rule was designed around the observed mistranslation `lobed shoes`, so a different English surface form was not corrected.

### P1

```text
Baseline: nuclei melt
Expected terminology: nuclei fuse
```

The same semantic terminology issue appeared in a different grammatical construction and did not match the development rule.

### E4

```text
Baseline: GPS cards
Expected terminology: GPS maps
```

The Italian source construction differed from the exact pattern expected by the rule.

These misses show the trade-off of the approach:

- narrow rules reduce accidental regressions
- narrow rules also miss valid corrections when either the source wording or mistranslation changes

The result of **0 observed regressions** applies only to this 20-sentence challenge set and is not a guarantee that the rule system can never introduce one.

## Human-Review Flag

One challenge case contained:

```text
piegatori → benders
```

in an origami context.

Because `piegatori` can describe bending/folding activities depending on context, the prototype does not automatically replace the term.

Instead it emits a review flag:

```text
piegatori_requires_activity_context:
inspect folding versus bending meaning
```

This is an example of preferring abstention over an unsafe automatic correction.

## Input-Perturbation Experiment

`test_perturbations.py`

The experiment tests whether small source changes can significantly alter translation.

Five perturbation types are used:

1. word/clause reordering
2. punctuation removal
3. capitalization changes
4. single-character misspellings
5. close-synonym substitution

Each category contains three original/perturbed pairs:

```text
15 pairs total
30 translated sentences
```

Run:

```bash
python code/test_perturbations.py
```

Generated output:

```text
data/perturbation_results.csv
```

The script leaves `semantic_effect` and `notes` blank for manual inspection.

### Observed Summary

Across the 15 pairs:

```text
Stable:             10
Minor change:        3
Significant change:  2
```

Reordering was stable in all three tested cases.

The strongest semantic failures came from capitalization and misspelling.

### Capitalization Example

```text
Rosa ha comprato un nuovo libro.
→ "Rosa bought a new book."

rosa ha comprato un nuovo libro.
→ "Pink bought a new book."
```

Lowercasing the proper name caused the translator to interpret `rosa` as the common-word meaning.

### Misspelling Example

```text
visitato → "visited"
vistato   → "saw"
```

A one-character typo changed the final English interpretation.

Overall, MarianMT was reasonably stable for most tested perturbations, but individual small input changes could still produce substantial semantic changes.

## Translation Reliability Without a Reference

The experiments motivate several signals that could be used when a verified English translation is unavailable:

- model uncertainty / low candidate confidence
- semantic instability under small meaning-preserving perturbations
- ambiguous or domain-specific terminology
- source/output consistency checks for names, numbers, dates, times, units, and similar details

A practical system could combine such evidence into a reliability score or directly flag severe inconsistencies for human review.

Fluency alone should not be treated as evidence of correctness: several observed outputs were grammatically plausible while changing the source meaning.

## Full Run Order

From `03_translation/`:

```bash
# Required only when FLORES+ CSVs are not already present
python code/download_flores.py

python code/baseline_translate.py
python code/build_eval_sample.py
python code/run_baseline_eval.py

python code/terminology_correct.py --self-test
python code/terminology_correct.py

python code/run_terminology_eval.py --dry-run
python code/run_terminology_eval.py

python code/test_perturbations.py
```

Several scripts intentionally refuse to overwrite result files. Move or delete the corresponding generated output before rerunning them.

## Main Takeaways

- Fluent English can still be semantically wrong.
- Translation errors often depend on domain and context rather than surface grammaticality.
- MarianMT was stable under most tested small perturbations, but capitalization and misspellings caused significant failures in individual cases.
- Source-aware terminology correction can achieve high precision when a known error pattern repeats.
- Exact pattern matching generalizes poorly when the same semantic error appears in a new surface form.
- Conservative automation reduces observed regressions at the cost of missed corrections.
- Human review remains important when the system detects ambiguity or unreliable behavior.
