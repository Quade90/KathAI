# 2.1 The Damaged Recording

## Folder layout
- `reference/` — clean reference recording (same passage used throughout)
- `degraded/` — versions with one physical/technical degradation applied at a time
- `restored/` — outputs of restoration techniques applied to degraded versions
- `code/` — one script per sub-question (see below)
- `plots/` — exported spectrograms / waveform / spectrum figures used in the write-up

## Code map (fill in as you build)
- `code/representations.py` — Part 1, Q1: waveform / spectrogram / frequency spectrum analysis
- `code/degrade.py` — Part 1, Q2–3: generate degraded versions, measure changes
- `code/restore.py` — Part 2, Q4–5: restoration techniques, before/after comparison
- `code/break_pipeline.py` — Part 2, Q6: restoration applied to wrong damage type
- `code/lossy_cleanup_wer.py` — Part 2, Q7: cleaner-sounding-but-lossy proof via ASR WER

## Summary of findings
_(fill in once done — what changed, what survived, what was lost, what could
be recovered, when restoration should not be trusted)_

## Key numbers
_(SNR / spectral distance / WER deltas — table once available)_
