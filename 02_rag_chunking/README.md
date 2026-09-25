# 2.2 The Chunk That Cut the Story in Half

## Folder layout
- `transcripts/raw/` — raw Whisper (or other ASR) output
- `transcripts/cleaned/` — lightly cleaned transcripts used for chunking
- `code/` — one script per sub-question (see below)
- `eval/` — hand-labeled query set + retrieval evaluation results
- `bonus/` — end-to-end queryable pipeline on real lecture(s)

## Code map (fill in as you build)
- `code/naive_chunking_demo.py` — Q1: demonstrates why fixed-size token chunking breaks on transcripts
- `code/strategy_semantic.py` — Q2: chunking strategy A
- `code/strategy_speaker_turn.py` — Q2: chunking strategy B (swap name if different strategy chosen)
- `code/retrieval_eval.py` — Q3: evaluation metric / experiment
- `bonus/transcribe.py` — Whisper wrapper
- `bonus/query.py` — end-to-end queryable pipeline
- `bonus/lectures_used.md` — which videos were used and why

## Summary of findings
_(fill in once done)_

## Eval results
_(retrieval hit-rate @ k, per chunking strategy — table once available)_
