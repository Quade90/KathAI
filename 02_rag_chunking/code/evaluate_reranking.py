from pathlib import Path
import json
import re

import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

INITIAL_K = 10

K_VALUES = [
    1,
    3,
    5,
]


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

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
# MODELS
# ============================================================

print(
    f"Loading embedding model: "
    f"{EMBEDDING_MODEL_NAME}"
)

embedding_model = (
    SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )
)

print(
    f"Loading reranker: "
    f"{RERANKER_MODEL_NAME}"
)

reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)


# ============================================================
# LOAD DATA
# ============================================================

embeddings = np.load(
    EMBEDDINGS_PATH
)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(file)

with open(
    GROUND_TRUTH_PATH,
    "r",
    encoding="utf-8",
) as file:

    ground_truth_data = json.load(
        file
    )


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):

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

    if "local_chunk_id" in chunk:
        return chunk[
            "local_chunk_id"
        ]

    return chunk["chunk_id"]


# ============================================================
# DENSE RETRIEVAL
# ============================================================

def dense_retrieve(
    question,
    candidate_count,
):

    query_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
    )

    scores = (
        embeddings
        @ query_embedding
    )

    ranking = np.argsort(
        scores
    )[::-1]

    ranking = ranking[
        :candidate_count
    ]

    results = []

    for rank, index in enumerate(
        ranking,
        start=1,
    ):

        chunk = metadata[index]

        results.append(
            {
                "rank": rank,
                "source": chunk[
                    "source"
                ],
                "chunk_id": (
                    get_chunk_id(
                        chunk
                    )
                ),
                "text": chunk[
                    "text"
                ],
                "dense_score": float(
                    scores[index]
                ),
            }
        )

    return results


# ============================================================
# RERANK
# ============================================================

def rerank(
    question,
    candidates,
):

    pairs = [
        (
            question,
            candidate["text"],
        )
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    results = []

    for candidate, score in zip(
        candidates,
        scores,
    ):

        result = candidate.copy()

        result[
            "reranker_score"
        ] = float(score)

        results.append(
            result
        )

    results.sort(
        key=lambda item:
        item["reranker_score"],
        reverse=True,
    )

    return results


# ============================================================
# CONTEXT COMPLETENESS CHECK
# ============================================================

def contains_ground_truth(
    results,
    source,
    ground_truth,
):

    # Only chunks from the correct
    # source can contribute.
    selected = [
        result
        for result in results
        if result["source"] == source
    ]

    # Restore transcript order rather
    # than retrieval ranking order.
    selected.sort(
        key=lambda item:
        item["chunk_id"]
    )

    context = " ".join(
        result["text"]
        for result in selected
    )

    normalized_context = (
        normalize_text(
            context
        )
    )

    normalized_gt = (
        normalize_text(
            ground_truth
        )
    )

    return (
        normalized_gt
        in normalized_context
    )


# ============================================================
# EVALUATION
# ============================================================

dense_scores = {
    k: []
    for k in K_VALUES
}

reranked_scores = {
    k: []
    for k in K_VALUES
}


print()
print("=" * 80)

print(
    "DENSE VS RERANKED "
    "SEMANTIC RETRIEVAL"
)

print("=" * 80)


for item in ground_truth_data:

    question = item[
        "question"
    ]

    source = item[
        "source"
    ]

    ground_truth = item[
        "ground_truth"
    ]

    # --------------------------------------------
    # Dense top 10
    # --------------------------------------------

    dense_candidates = (
        dense_retrieve(
            question,
            INITIAL_K,
        )
    )

    # --------------------------------------------
    # Cross-encoder ranking
    # --------------------------------------------

    reranked_candidates = rerank(
        question,
        dense_candidates,
    )

    print()
    print(
        f"Question: "
        f"{question}"
    )

    for k in K_VALUES:

        dense_success = (
            contains_ground_truth(
                dense_candidates[:k],
                source,
                ground_truth,
            )
        )

        reranked_success = (
            contains_ground_truth(
                reranked_candidates[:k],
                source,
                ground_truth,
            )
        )

        dense_scores[k].append(
            int(dense_success)
        )

        reranked_scores[k].append(
            int(reranked_success)
        )

        print(
            f"  Recall@{k}: "
            f"Dense="
            f"{int(dense_success)} "
            f"Reranked="
            f"{int(reranked_success)}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 80)

print(
    "FINAL RESULTS"
)

print("=" * 80)


for k in K_VALUES:

    dense_recall = float(
        np.mean(
            dense_scores[k]
        )
    )

    reranked_recall = float(
        np.mean(
            reranked_scores[k]
        )
    )

    difference = (
        reranked_recall
        - dense_recall
    )

    print()
    print(
        f"Recall@{k}"
    )

    print(
        f"Dense semantic: "
        f"{dense_recall:.3f}"
    )

    print(
        f"Reranked semantic: "
        f"{reranked_recall:.3f}"
    )

    print(
        f"Difference: "
        f"{difference:+.3f}"
    )


print()
print("=" * 80)