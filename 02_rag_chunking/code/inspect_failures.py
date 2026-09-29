from pathlib import Path
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

K = 3
SHOW_TOP = 10


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "embeddings"
    / "semantic_chunk_embeddings.npy"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "embeddings"
    / "semantic_chunk_metadata.json"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "ground_truth_30.json"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    """
    Same basic normalization idea as compare_chunking.py:
    lowercase, remove punctuation, collapse whitespace.
    """

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def get_chunk_id(chunk):
    """
    Support both metadata formats.
    """

    if "local_chunk_id" in chunk:
        return chunk["local_chunk_id"]

    return chunk["chunk_id"]


def retrieve_all(
    question,
    model,
    embeddings,
    metadata,
):
    """
    Rank every semantic chunk by cosine similarity.

    Embeddings are already normalized, so dot product
    equals cosine similarity.
    """

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = embeddings @ query_embedding

    ranking = np.argsort(scores)[::-1]

    results = []

    for rank, index in enumerate(
        ranking,
        start=1,
    ):

        chunk = metadata[index]

        results.append(
            {
                "rank": rank,
                "index": int(index),
                "score": float(scores[index]),
                "source": chunk["source"],
                "chunk_id": get_chunk_id(chunk),
                "token_count": chunk.get(
                    "token_count",
                    None,
                ),
                "text": chunk["text"],
            }
        )

    return results


def find_required_chunks(
    source,
    ground_truth,
    metadata,
):
    """
    Determine which semantic chunks contain the exact
    ground-truth passage.

    We reconstruct the source from its semantic chunks,
    find the normalized ground-truth span, then determine
    which chunks intersect that span.
    """

    source_chunks = [
        chunk
        for chunk in metadata
        if chunk["source"] == source
    ]

    source_chunks.sort(
        key=get_chunk_id
    )

    pieces = []

    chunk_spans = []

    current_position = 0

    for chunk in source_chunks:

        normalized_chunk = normalize_text(
            chunk["text"]
        )

        if pieces:
            current_position += 1

        start = current_position

        pieces.append(
            normalized_chunk
        )

        current_position += len(
            normalized_chunk
        )

        end = current_position

        chunk_spans.append(
            {
                "chunk_id": get_chunk_id(
                    chunk
                ),
                "start": start,
                "end": end,
            }
        )

    reconstructed = " ".join(
        pieces
    )

    normalized_gt = normalize_text(
        ground_truth
    )

    gt_start = reconstructed.find(
        normalized_gt
    )

    if gt_start == -1:
        return None

    gt_end = (
        gt_start
        + len(normalized_gt)
    )

    required_chunks = []

    for chunk in chunk_spans:

        # Check whether the chunk overlaps
        # the ground-truth character span.
        overlaps = (
            chunk["start"] < gt_end
            and
            chunk["end"] > gt_start
        )

        if overlaps:
            required_chunks.append(
                chunk["chunk_id"]
            )

    return required_chunks


def get_required_ranks(
    source,
    required_chunks,
    retrieval_results,
):
    """
    Find where each required semantic chunk appeared
    in the retrieval ranking.
    """

    required_ranks = []

    for chunk_id in required_chunks:

        result = next(
            (
                result
                for result
                in retrieval_results
                if (
                    result["source"] == source
                    and
                    result["chunk_id"]
                    == chunk_id
                )
            ),
            None,
        )

        if result is not None:

            required_ranks.append(
                {
                    "chunk_id": chunk_id,
                    "rank": result["rank"],
                    "score": result["score"],
                }
            )

    return required_ranks


# ============================================================
# LOAD DATA
# ============================================================

print(
    f"Loading embedding model: "
    f"{MODEL_NAME}"
)

model = SentenceTransformer(
    MODEL_NAME
)

embeddings = np.load(
    EMBEDDINGS_PATH
)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(
        file
    )

with open(
    GROUND_TRUTH_PATH,
    "r",
    encoding="utf-8",
) as file:

    ground_truth_data = json.load(
        file
    )


# ============================================================
# ANALYSIS
# ============================================================

failures = []


for item in ground_truth_data:

    question = item["question"]
    source = item["source"]
    ground_truth = item["ground_truth"]

    required_chunks = (
        find_required_chunks(
            source,
            ground_truth,
            metadata,
        )
    )

    if required_chunks is None:

        print()
        print(
            "WARNING: Could not locate "
            "ground truth:"
        )

        print(question)

        continue

    retrieval_results = retrieve_all(
        question,
        model,
        embeddings,
        metadata,
    )

    required_ranks = (
        get_required_ranks(
            source,
            required_chunks,
            retrieval_results,
        )
    )

    ranks = [
        result["rank"]
        for result in required_ranks
    ]

    # The ground truth is completely available
    # within top-K only if every required chunk
    # appears within top-K.
    success_at_k = (
        len(ranks)
        == len(required_chunks)
        and
        all(
            rank <= K
            for rank in ranks
        )
    )

    if success_at_k:
        continue

    # --------------------------------------------------------
    # CLASSIFY FAILURE
    # --------------------------------------------------------

    if len(required_chunks) > K:

        classification = (
            "BOUNDARY FRAGMENTATION"
        )

        explanation = (
            f"The answer spans "
            f"{len(required_chunks)} semantic "
            f"chunks, so retrieving only "
            f"{K} chunks cannot recover the "
            f"whole passage."
        )

    elif (
        ranks
        and
        max(ranks) <= 5
    ):

        classification = (
            "NEAR-MISS RANKING ISSUE"
        )

        explanation = (
            "All required chunks exist and "
            "are ranked within the top 5, "
            "but at least one falls outside "
            "the top 3."
        )

    else:

        classification = (
            "RETRIEVAL RANKING ISSUE"
        )

        explanation = (
            "The answer requires at most "
            f"{K} chunks, but one or more "
            "required chunks are ranked "
            "too low."
        )

    failures.append(
        {
            "question": question,
            "source": source,
            "required_chunks": required_chunks,
            "required_ranks": required_ranks,
            "classification": classification,
            "explanation": explanation,
            "retrieval_results": retrieval_results,
        }
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print(
    "=" * 80
)

print(
    f"SEMANTIC RECALL@{K} FAILURE ANALYSIS"
)

print(
    "=" * 80
)

print(
    f"\nTotal failures: "
    f"{len(failures)}"
)


for number, failure in enumerate(
    failures,
    start=1,
):

    print()
    print(
        "=" * 80
    )

    print(
        f"FAILURE {number}"
    )

    print(
        "=" * 80
    )

    print(
        f"\nQuestion:\n"
        f"{failure['question']}"
    )

    print(
        f"\nSource: "
        f"{failure['source']}"
    )

    print(
        f"\nClassification: "
        f"{failure['classification']}"
    )

    print(
        f"\nReason:\n"
        f"{failure['explanation']}"
    )

    print(
        "\nRequired semantic chunks:"
    )

    print(
        failure[
            "required_chunks"
        ]
    )

    print(
        "\nRanks of required chunks:"
    )

    for result in failure[
        "required_ranks"
    ]:

        print(
            f"  Chunk "
            f"{result['chunk_id']}: "
            f"rank {result['rank']}, "
            f"score "
            f"{result['score']:.4f}"
        )

    print(
        f"\nTop {SHOW_TOP} retrieved chunks:"
    )

    for result in failure[
        "retrieval_results"
    ][:SHOW_TOP]:

        marker = ""

        if (
            result["source"]
            == failure["source"]
            and
            result["chunk_id"]
            in failure[
                "required_chunks"
            ]
        ):
            marker = " <-- REQUIRED"

        print(
            f"  Rank "
            f"{result['rank']:2d} | "
            f"{result['source']:12s} | "
            f"chunk "
            f"{result['chunk_id']:3d} | "
            f"score "
            f"{result['score']:.4f}"
            f"{marker}"
        )


# ============================================================
# SUMMARY
# ============================================================

fragmentation_count = sum(
    failure["classification"]
    == "BOUNDARY FRAGMENTATION"
    for failure in failures
)

near_miss_count = sum(
    failure["classification"]
    == "NEAR-MISS RANKING ISSUE"
    for failure in failures
)

ranking_count = sum(
    failure["classification"]
    == "RETRIEVAL RANKING ISSUE"
    for failure in failures
)


print()
print(
    "=" * 80
)

print(
    "SUMMARY"
)

print(
    "=" * 80
)

print(
    f"Boundary fragmentation : "
    f"{fragmentation_count}"
)

print(
    f"Near-miss ranking       : "
    f"{near_miss_count}"
)

print(
    f"Other ranking issues    : "
    f"{ranking_count}"
)