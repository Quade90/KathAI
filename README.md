# KathAI Project Member Application

This repository contains the implementation for the technical portion of the KathAI Project Member Application.

The work is split into three independent investigations:

1. **Audio restoration and degradation analysis**
2. **Semantic chunking and retrieval for long-form transcripts**
3. **Translation reliability and terminology-aware correction**

Each section contains its own README with the implementation details, experiment design, run order, results, and limitations.

## Repository Structure

```text
KathAI/
├── README.md
├── requirements.txt
├── LICENSE
├── 01_audio_restoration/
│   ├── README.md
│   └── code/
├── 02_rag_chunking/
│   ├── README.md
│   ├── code/
│   ├── evaluation/
│   │   └── ground_truth_30.json
│   └── transcripts/
│       └── raw/
│           ├── 3b1b.txt
│           ├── neo.txt
│           └── veritasium.txt
└── 03_translation/
    ├── README.md
    ├── code/
    └── data/
        ├── terminology_challenge.csv
        └── perturbation_challenge.csv
```

Generated files, downloaded datasets, model caches, plots, audio outputs, embeddings, working notes, and the application write-up are intentionally excluded from Git.

## Setup

Create and activate a Python virtual environment, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

The project uses:

- NumPy / SciPy / pandas
- librosa / SoundFile / Matplotlib
- OpenAI Whisper
- sentence-transformers / spaCy
- PyTorch / Transformers / SentencePiece
- Hugging Face datasets

### FFmpeg

FFmpeg is a system dependency used by the audio-compression workflow and by Whisper when regenerating transcripts.

Install FFmpeg separately and make sure it is available on your system. Some scripts currently contain a local Windows FFmpeg path, so update that path if your installation is elsewhere.

## Section 2.1 — Audio Restoration

[`01_audio_restoration/README.md`](01_audio_restoration/README.md)

This section investigates how different degradations alter speech in the time and frequency domains and when restoration can or cannot recover the original information.

Implemented degradations include:

- 50 Hz electrical hum
- 500–4000 Hz bandpass filtering
- hard clipping
- synthetic reverberation
- 32 kbps MP3 compression

Restoration experiments include automatic hum detection with notch filtering, regularized inverse-filter dereverberation, deliberately applying the wrong restoration model, and a pitch-tracking experiment showing that cleaner-sounding audio can still lose useful information.

## Section 2.2 — Semantic Chunking and Retrieval

[`02_rag_chunking/README.md`](02_rag_chunking/README.md)

This section builds a retrieval pipeline over three technical-video transcripts.

The system progresses through:

- fixed 512-token chunking baseline
- sentence-level semantic boundary detection
- similarity smoothing and local-minimum boundary selection
- minimum/maximum token constraints
- BGE dense retrieval
- out-of-domain rejection
- cross-encoder reranking

The final implementation improves context-complete Recall@1 from **0.400** for the fixed-size baseline to **0.700** with refined semantic chunking and cross-encoder reranking.

## Section 2.3 — Translation Reliability

[`03_translation/README.md`](03_translation/README.md)

This section studies semantic failures in Italian-to-English translation using:

- `Helsinki-NLP/opus-mt-it-en`
- FLORES+
- controlled perturbation tests
- a conservative source-aware terminology correction layer

The terminology intervention is tested on a separate 20-sentence challenge set. It makes 3 automatic corrections, all 3 of which are correct in the test set, flags 1 additional case for review, and introduces 0 observed regressions.

## Reproducibility

Each section README contains the exact run order.

Most generated artifacts are deliberately excluded from version control. The repository keeps the code and small experiment-defining inputs while allowing results to be regenerated locally.

Examples of ignored artifacts include:

- degraded/restored audio and plots
- embeddings and generated chunks
- downloaded FLORES+ files
- generated evaluation CSVs
- model/library caches
- local notes and write-up files

## License

See [`LICENSE`](LICENSE).
