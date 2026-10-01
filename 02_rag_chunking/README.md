# 2.2 — The Chunk That Cut the Story in Half

This section investigates how chunk boundaries affect retrieval from long speech transcripts and builds a small semantic RAG retrieval pipeline over three technical-video transcripts.

The central problem is that fixed token windows can split an explanation across chunk boundaries, causing retrieval to return only part of the context required to answer a question.

## Dataset

Three English transcripts are used:

```text
transcripts/raw/
├── 3b1b.txt
├── neo.txt
└── veritasium.txt
```

They cover different technical/informational topics:

- mathematical visual proofs
- the U.S. presidential motorcade
- zipper design and mechanics

The transcripts were originally produced with OpenAI Whisper.

Raw transcript text is kept in the repository so that the chunking and retrieval experiments can be reproduced without rerunning transcription.

## Code

```text
code/
├── transcribe.py
├── baseline_pipeline.py
├── semantic_pipeline.py
├── identify_threshold.py
├── compare_chunking.py
├── inspect_failures.py
├── evaluate_reranking.py
└── test_retrieve.py
```

Generated chunk files, embedding matrices, and evaluation-output CSVs are not committed.

The manually constructed ground-truth evaluation set is kept:

```text
evaluation/ground_truth_30.json
```

## Models

### Dense Embedding Model

```text
BAAI/bge-small-en-v1.5
```

The same embedding model is used for:

- sentence embeddings during semantic boundary detection
- chunk embeddings
- query embeddings
- dense retrieval

Embeddings are normalized, so the dot product is equivalent to cosine similarity.

### Cross-Encoder Reranker

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The reranker is used only after dense retrieval to reorder the top candidate chunks.

## Optional Transcript Generation

`transcribe.py` uses OpenAI Whisper:

```text
Whisper model: small
Language: English
```

It scans `videos/` for MP3 files and writes transcripts to `transcripts/raw/`.

The repository already contains the raw text transcripts, so this step is optional.

If regenerating them:

```bash
python code/transcribe.py
```

> `transcribe.py` contains a local Windows FFmpeg path. Change it if your FFmpeg installation is elsewhere.

## Fixed-Size Baseline

`baseline_pipeline.py`

The baseline ignores semantic structure and divides each transcript into non-overlapping blocks of:

```text
512 tokenizer tokens
```

Each chunk is embedded using BGE and stored with source and token-position metadata.

Run:

```bash
python code/baseline_pipeline.py
```

## Semantic Chunking

`semantic_pipeline.py`

The semantic pipeline uses sentences as atomic units.

### 1. Sentence Segmentation

spaCy's English sentencizer divides each transcript into sentences.

### 2. Adjacent Semantic Similarity

Each sentence is embedded using BGE.

For adjacent sentence embeddings, cosine similarity is calculated. A low similarity indicates a possible context transition.

### 3. Similarity Smoothing

The similarity sequence is smoothed with:

```text
window size = 3
```

This suppresses small local fluctuations before boundary detection.

### 4. Candidate Boundary Threshold

Candidate semantic boundaries must fall at or below the:

```text
5th percentile
```

of the smoothed adjacent-sentence similarity distribution.

The 5th percentile was selected after testing 20%, 10%, 5%, and 2.5%.

Higher percentiles created too many small fragments, while 2.5% produced too few semantic boundaries.

### 5. Local Minimum Requirement

A candidate must also be a local minimum in the smoothed similarity profile.

This chooses the bottom of a similarity valley rather than the first point at which similarity starts decreasing.

### 6. Token Constraints

Minimum chunk size:

```text
100 tokens
```

Maximum chunk size:

```text
model.max_seq_length - tokenizer special tokens
```

The maximum is therefore based on the actual usable sequence length of the embedding model rather than assuming that every chunk can contain 512 ordinary text tokens.

### 7. Merging Small Chunks

Chunks below the minimum size are merged with a valid neighboring chunk.

If both directions are possible, the system compares semantic similarity with the left and right neighbors and merges toward the more similar one.

Run:

```bash
python code/semantic_pipeline.py
```

## Relevance Threshold

`identify_threshold.py`

A separate test was performed to determine when the dense retriever should reject a query as out-of-domain.

The test contains:

- 18 in-domain queries
- 6 borderline queries
- 8 unrelated queries

### Similarity Distribution

| Query type | Count | Mean | Min | Max |
|---|---:|---:|---:|---:|
| In-domain | 18 | 0.7410 | 0.5445 | 0.8496 |
| Borderline | 6 | 0.6866 | 0.5774 | 0.7862 |
| Unrelated | 8 | 0.4957 | 0.4118 | 0.5528 |

The selected BGE cosine-similarity threshold is:

```text
0.60
```

This rejects all unrelated questions in the tested set, at the cost of rejecting some lower-scoring relevant/borderline queries.

Run:

```bash
python code/identify_threshold.py
```

The threshold applies to the **dense BGE score**, not the cross-encoder score.

## Evaluation Method

`compare_chunking.py`

A manually constructed evaluation set contains:

```text
30 questions
10 questions per transcript
```

Some questions deliberately require context that can cross a chunk boundary.

For each question, the system retrieves the top k chunks for:

```text
k = 1, 3, 5
```

A retrieval is counted as successful only if the retrieved chunks collectively contain the complete ground-truth passage required to answer the question.

The same embedding model, questions, retrieval method, and k values are used for both the fixed-size and semantic pipelines.

The 0.60 out-of-domain threshold is not used during the chunking comparison.

Run:

```bash
python code/compare_chunking.py
```

## Chunking Results

### Initial Semantic Implementation

| Method | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| Fixed 512-token baseline | 0.400 | 0.933 | 0.933 |
| Initial semantic chunking | 0.367 | 0.667 | 0.800 |

Simply introducing semantic similarity was not enough. The initial semantic pipeline performed worse than the fixed-size baseline.

### Refined Semantic Implementation

After adding smoothing, a 5th-percentile cutoff, local-minimum boundary selection, and chunk-size constraints:

| Method | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| Fixed 512-token baseline | 0.400 | 0.933 | 0.933 |
| Refined semantic chunking | 0.667 | 0.867 | 1.000 |

The refined semantic pipeline substantially improves Recall@1 and reaches complete context retrieval at Recall@5, although it remains slightly below the fixed-size baseline at Recall@3.

## Failure Analysis

`inspect_failures.py`

The four failed semantic Recall@3 cases were inspected in detail.

Failures are classified as:

- boundary fragmentation
- near-miss ranking issue
- retrieval ranking issue

All four observed Recall@3 failures were **near-miss ranking issues**:

- the required chunks existed
- the required context was within the top 5
- at least one required chunk fell outside the top 3

This motivated improving ranking rather than changing the chunking pipeline again.

Run:

```bash
python code/inspect_failures.py
```

## Cross-Encoder Reranking

`evaluate_reranking.py`

The reranking experiment works as follows:

1. retrieve the top 10 chunks using BGE
2. pair each candidate with the query
3. score each pair using the cross-encoder
4. reorder the candidates
5. evaluate context-complete Recall@1, Recall@3, and Recall@5

### Results

| Retrieval method | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| Dense semantic retrieval | 0.667 | 0.867 | 1.000 |
| + cross-encoder reranking | 0.700 | 0.900 | 1.000 |

Reranking improves both Recall@1 and Recall@3 while preserving Recall@5.

Run:

```bash
python code/evaluate_reranking.py
```

## Final Interactive Retrieval Pipeline

`test_retrieve.py`

The final retrieval system uses:

```text
Dense embedding model: BAAI/bge-small-en-v1.5
Initial dense candidates: 10
Dense relevance threshold: 0.60
Reranker: cross-encoder/ms-marco-MiniLM-L6-v2
Final returned chunks: 5
```

Flow:

```text
Question
   ↓
BGE query embedding
   ↓
Dense cosine-similarity retrieval
   ↓
Top-10 candidates
   ↓
Reject if top dense score < 0.60
   ↓
Cross-encoder reranking
   ↓
Top-5 transcript passages
```

Run:

```bash
python code/test_retrieve.py
```

Then enter natural-language questions at the prompt.

Example:

```text
Why do so many zippers have YKK written on them?
```

Enter `exit` or `quit` to stop.

## Full Run Order

From `02_rag_chunking/`:

```bash
# Optional: raw transcripts are already included
python code/transcribe.py

python code/baseline_pipeline.py
python code/semantic_pipeline.py
python code/identify_threshold.py
python code/compare_chunking.py
python code/inspect_failures.py
python code/evaluate_reranking.py
python code/test_retrieve.py
```

## Final Progression

| Implementation | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| Fixed baseline | 0.400 | 0.933 | 0.933 |
| Initial semantic | 0.367 | 0.667 | 0.800 |
| Refined semantic | 0.667 | 0.867 | 1.000 |
| Refined semantic + cross-encoder | 0.700 | 0.900 | 1.000 |

## Limitations

- The evaluation set contains only 30 manually constructed questions across three transcripts.
- The selected relevance threshold was derived from only 32 test queries.
- Results therefore describe this experiment and should not be assumed to generalize to a 40,000-hour archive without further testing.
- Semantic boundaries do not automatically outperform well-sized fixed blocks at every retrieval depth.
- Transcription errors remain a source of noise before chunking even begins.

## Main Takeaways

- Fixed-size chunks can split context even when retrieval similarity is high.
- Semantic boundaries help most when paired with sensible token constraints and smoothing.
- A semantic chunker can still lose to a simple baseline at particular retrieval depths.
- Failure analysis is important: the remaining Recall@3 failures were ranking problems, not observed boundary-fragmentation failures.
- A two-stage dense-retrieval + cross-encoder pipeline improved the final ranking quality.
