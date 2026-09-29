from pathlib import Path
import json

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BOUNDARY_PERCENTILE = 5.0

MIN_CHUNK_TOKENS = 100

SMOOTHING_WINDOW = 3


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_TRANSCRIPT_DIR = PROJECT_ROOT / "transcripts" / "raw"
CHUNK_OUTPUT_DIR = PROJECT_ROOT / "transcripts" / "chunks"
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"

CHUNK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = EMBEDDING_DIR / "semantic_chunk_embeddings.npy"
METADATA_PATH = EMBEDDING_DIR / "semantic_chunk_metadata.json"


# ============================================================
# LOAD MODEL + SENTENCE SPLITTER
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer

special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

MAX_CHUNK_TOKENS = (
    model.max_seq_length - special_tokens
)

print(
    f"Maximum usable chunk tokens: "
    f"{MAX_CHUNK_TOKENS}"
)

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


# ============================================================
# HELPERS
# ============================================================

def split_into_sentences(text):

    doc = nlp(text)

    sentences = [
        sentence.text.strip()
        for sentence in doc.sents
        if sentence.text.strip()
    ]

    return sentences


def token_count(text):

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )


def smooth_similarities(
    similarities,
    window=3,
):

    similarities = np.asarray(
        similarities,
        dtype=np.float32,
    )

    if (
        len(similarities) < 3
        or window <= 1
    ):
        return similarities.copy()

    radius = window // 2

    smoothed = np.empty_like(
        similarities
    )

    for i in range(
        len(similarities)
    ):

        left = max(
            0,
            i - radius,
        )

        right = min(
            len(similarities),
            i + radius + 1,
        )

        smoothed[i] = np.mean(
            similarities[left:right]
        )

    return smoothed


def find_semantic_boundaries(
    sentences,
):

    if len(sentences) <= 1:

        return (
            set(),
            np.array([]),
            np.array([]),
            None,
        )

    sentence_embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    raw_similarities = np.sum(
        sentence_embeddings[:-1]
        * sentence_embeddings[1:],
        axis=1,
    )

    smoothed_similarities = (
        smooth_similarities(
            raw_similarities,
            SMOOTHING_WINDOW,
        )
    )

    threshold = np.percentile(
        smoothed_similarities,
        BOUNDARY_PERCENTILE,
    )

    boundary_indices = set()

    for i, similarity in enumerate(
        smoothed_similarities
    ):

        if similarity > threshold:
            continue

        left_similarity = (
            smoothed_similarities[i - 1]
            if i > 0
            else np.inf
        )

        right_similarity = (
            smoothed_similarities[i + 1]
            if i
            < len(
                smoothed_similarities
            ) - 1
            else np.inf
        )

        is_local_minimum = (
            similarity
            <= left_similarity
            and
            similarity
            <= right_similarity
        )

        if is_local_minimum:

            boundary_indices.add(i)

    return (
        boundary_indices,
        raw_similarities,
        smoothed_similarities,
        threshold,
    )


def create_initial_chunks(
    sentences,
    boundary_indices,
):

    chunks = []

    current_sentences = []
    current_tokens = 0

    for i, sentence in enumerate(
        sentences
    ):

        sentence_tokens = (
            token_count(sentence)
        )

        if (
            current_sentences
            and
            current_tokens
            + sentence_tokens
            > MAX_CHUNK_TOKENS
        ):

            chunks.append(
                " ".join(
                    current_sentences
                ).strip()
            )

            current_sentences = []
            current_tokens = 0

        current_sentences.append(
            sentence
        )

        current_tokens += (
            sentence_tokens
        )

        if (
            i in boundary_indices
            and
            current_tokens
            >= MIN_CHUNK_TOKENS
        ):

            chunks.append(
                " ".join(
                    current_sentences
                ).strip()
            )

            current_sentences = []
            current_tokens = 0

    if current_sentences:

        chunks.append(
            " ".join(
                current_sentences
            ).strip()
        )

    return chunks


def merge_small_chunks(chunks):

    if len(chunks) <= 1:
        return chunks

    chunks = chunks.copy()

    changed = True

    while changed:

        changed = False

        for i in range(
            len(chunks)
        ):

            if (
                token_count(chunks[i])
                >= MIN_CHUNK_TOKENS
            ):
                continue

            candidates = []

            # ------------------------------------
            # LEFT MERGE
            # ------------------------------------

            if i > 0:

                merged_text = (
                    chunks[i - 1]
                    + " "
                    + chunks[i]
                )

                if (
                    token_count(
                        merged_text
                    )
                    <= MAX_CHUNK_TOKENS
                ):

                    candidates.append(
                        (
                            "left",
                            merged_text,
                        )
                    )

            # ------------------------------------
            # RIGHT MERGE
            # ------------------------------------

            if (
                i
                < len(chunks) - 1
            ):

                merged_text = (
                    chunks[i]
                    + " "
                    + chunks[i + 1]
                )

                if (
                    token_count(
                        merged_text
                    )
                    <= MAX_CHUNK_TOKENS
                ):

                    candidates.append(
                        (
                            "right",
                            merged_text,
                        )
                    )

            if not candidates:
                continue

            # ------------------------------------
            # ONLY ONE VALID DIRECTION
            # ------------------------------------

            if len(candidates) == 1:

                direction, merged_text = (
                    candidates[0]
                )

            else:

                current_embedding = (
                    model.encode(
                        [chunks[i]],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )[0]
                )

                left_embedding = (
                    model.encode(
                        [chunks[i - 1]],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )[0]
                )

                right_embedding = (
                    model.encode(
                        [chunks[i + 1]],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )[0]
                )

                left_similarity = np.dot(
                    current_embedding,
                    left_embedding,
                )

                right_similarity = np.dot(
                    current_embedding,
                    right_embedding,
                )

                if (
                    left_similarity
                    >= right_similarity
                ):

                    direction = "left"

                    merged_text = (
                        chunks[i - 1]
                        + " "
                        + chunks[i]
                    )

                else:

                    direction = "right"

                    merged_text = (
                        chunks[i]
                        + " "
                        + chunks[i + 1]
                    )

            # ------------------------------------
            # PERFORM MERGE
            # ------------------------------------

            if direction == "left":

                chunks[i - 1] = (
                    merged_text
                )

                del chunks[i]

            else:

                chunks[i] = (
                    merged_text
                )

                del chunks[i + 1]

            changed = True
            break

    return chunks


def semantic_chunk_transcript(text):

    sentences = split_into_sentences(
        text
    )

    print(
        f"  Sentences: "
        f"{len(sentences)}"
    )

    (
        boundary_indices,
        raw_similarities,
        smoothed_similarities,
        threshold,
    ) = find_semantic_boundaries(
        sentences
    )

    if threshold is not None:

        print(
            f"  Similarity threshold "
            f"({BOUNDARY_PERCENTILE}th percentile): "
            f"{threshold:.4f}"
        )

    print(
        f"  Candidate semantic boundaries: "
        f"{len(boundary_indices)}"
    )

    chunks = create_initial_chunks(
        sentences,
        boundary_indices,
    )

    print(
        f"  Initial chunks: "
        f"{len(chunks)}"
    )

    chunks = merge_small_chunks(
        chunks
    )

    print(
        f"  Final chunks: "
        f"{len(chunks)}"
    )

    return chunks


# ============================================================
# PROCESS ALL TRANSCRIPTS
# ============================================================

all_chunk_texts = []
all_metadata = []

transcript_files = sorted(
    RAW_TRANSCRIPT_DIR.glob(
        "*.txt"
    )
)

if not transcript_files:

    raise FileNotFoundError(
        f"No transcript .txt files found in:\n"
        f"{RAW_TRANSCRIPT_DIR}"
    )


print()

print(
    "=" * 70
)

print(
    "SEMANTIC CHUNKING"
)

print(
    "=" * 70
)


for transcript_path in transcript_files:

    source = transcript_path.stem

    print()

    print(
        f"Processing: {source}"
    )

    print(
        "-" * 70
    )

    text = (
        transcript_path
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    chunks = (
        semantic_chunk_transcript(
            text
        )
    )

    chunk_records = []
    token_sizes = []

    for (
        chunk_id,
        chunk_text,
    ) in enumerate(chunks):

        tokens = token_count(
            chunk_text
        )

        token_sizes.append(
            tokens
        )

        record = {

            "source": source,

            # Used in readable chunk files
            "chunk_id": chunk_id,

            # Used by compare_chunking.py
            "local_chunk_id": chunk_id,

            "token_count": tokens,

            "text": chunk_text,
        }

        chunk_records.append(
            record
        )

        all_chunk_texts.append(
            chunk_text
        )

        all_metadata.append(
            record
        )


    chunk_output_path = (
        CHUNK_OUTPUT_DIR
        / f"{source}_chunks.json"
    )

    chunk_output_path.write_text(
        json.dumps(
            chunk_records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    print()

    print(
        "  Chunk statistics:"
    )

    if token_sizes:

        print(
            f"    Minimum tokens : "
            f"{min(token_sizes)}"
        )

        print(
            f"    Maximum tokens : "
            f"{max(token_sizes)}"
        )

        print(
            f"    Average tokens : "
            f"{np.mean(token_sizes):.1f}"
        )

    print(
        f"  Saved chunks -> "
        f"{chunk_output_path}"
    )


# ============================================================
# EMBED FINAL CHUNKS
# ============================================================

print()

print(
    "=" * 70
)

print(
    "EMBEDDING FINAL SEMANTIC CHUNKS"
)

print(
    "=" * 70
)


chunk_embeddings = model.encode(
    all_chunk_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

chunk_embeddings = np.asarray(
    chunk_embeddings,
    dtype=np.float32,
)


np.save(
    EMBEDDINGS_PATH,
    chunk_embeddings,
)


METADATA_PATH.write_text(
    json.dumps(
        all_metadata,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print(
    "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)


print(
    f"Boundary percentile : "
    f"{BOUNDARY_PERCENTILE}%"
)

print(
    f"Minimum chunk size  : "
    f"{MIN_CHUNK_TOKENS} tokens"
)

print(
    f"Maximum chunk size  : "
    f"{MAX_CHUNK_TOKENS} tokens"
)

print(
    f"Smoothing window    : "
    f"{SMOOTHING_WINDOW}"
)

print(
    f"Total chunks        : "
    f"{len(all_chunk_texts)}"
)


print()

print(
    f"Embeddings saved -> "
    f"{EMBEDDINGS_PATH}"
)

print(
    f"Metadata saved   -> "
    f"{METADATA_PATH}"
)