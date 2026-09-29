from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRANSCRIPTS_DIR = BASE_DIR / "transcripts" / "raw"
CHUNKS_DIR = BASE_DIR / "transcripts" / "baseline_chunks"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

FIXED_CHUNK_SIZE = 512


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

tokenizer = model.tokenizer


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    return " ".join(text.split())


def create_fixed_chunks(text):
    """
    Split transcript into fixed-size token chunks.
    No semantic awareness.
    """

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    chunks = []

    for start in range(0, len(token_ids), FIXED_CHUNK_SIZE):

        end = start + FIXED_CHUNK_SIZE

        chunk_tokens = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        chunks.append({
            "text": chunk_text.strip(),
            "token_start": start,
            "token_end": min(end, len(token_ids)),
            "token_count": len(chunk_tokens)
        })

    return chunks


# ============================================================
# PROCESS ONE TRANSCRIPT
# ============================================================

def process_transcript(transcript_path):

    print("=" * 60)
    print(f"Processing: {transcript_path.name}")

    with open(
        transcript_path,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    text = clean_text(text)

    chunks = create_fixed_chunks(text)

    print(f"Fixed chunks created: {len(chunks)}")

    for chunk_id, chunk in enumerate(chunks):

        chunk["chunk_id"] = chunk_id
        chunk["source"] = transcript_path.stem

    output_path = (
        CHUNKS_DIR /
        f"{transcript_path.stem}_baseline_chunks.json"
    )

    output_data = {
        "source": transcript_path.stem,
        "embedding_model": MODEL_NAME,
        "chunking_method": "fixed_size",
        "fixed_chunk_size": FIXED_CHUNK_SIZE,
        "number_of_chunks": len(chunks),
        "chunks": chunks
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Chunks saved -> {output_path.name}")

    # --------------------------------------------------------
    # Embed final baseline chunks
    # --------------------------------------------------------

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Generating baseline chunk embeddings...")

    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print()

    return chunks, chunk_embeddings


# ============================================================
# MAIN
# ============================================================

def main():

    transcript_files = sorted(
        TRANSCRIPTS_DIR.glob("*.txt")
    )

    print(
        f"Found {len(transcript_files)} transcript(s).\n"
    )

    if not transcript_files:

        raise FileNotFoundError(
            f"No transcript files found in:\n"
            f"{TRANSCRIPTS_DIR}"
        )

    all_chunks = []
    all_embeddings = []

    for transcript_path in transcript_files:

        chunks, embeddings = process_transcript(
            transcript_path
        )

        all_chunks.extend(chunks)

        if len(embeddings) > 0:
            all_embeddings.append(embeddings)

    if not all_embeddings:
        raise RuntimeError(
            "No baseline embeddings were generated."
        )

    combined_embeddings = np.vstack(
        all_embeddings
    )

    embeddings_path = (
        EMBEDDINGS_DIR /
        "baseline_chunk_embeddings.npy"
    )

    np.save(
        embeddings_path,
        combined_embeddings
    )

    metadata = []

    for global_id, chunk in enumerate(all_chunks):

        metadata.append({
            "global_chunk_id": global_id,
            "source": chunk["source"],
            "local_chunk_id": chunk["chunk_id"],
            "token_start": chunk["token_start"],
            "token_end": chunk["token_end"],
            "token_count": chunk["token_count"],
            "text": chunk["text"]
        })

    metadata_path = (
        EMBEDDINGS_DIR /
        "baseline_chunk_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 60)
    print("Baseline pipeline complete.")
    print(f"Total chunks: {len(all_chunks)}")
    print(
        f"Embedding matrix shape: "
        f"{combined_embeddings.shape}"
    )
    print(f"Embeddings -> {embeddings_path}")
    print(f"Metadata   -> {metadata_path}")


if __name__ == "__main__":
    main()