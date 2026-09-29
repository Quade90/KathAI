from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

# Number of candidates obtained from dense retrieval
INITIAL_K = 10

# Final number shown / sent to the LLM
FINAL_K = 5

# Our previously measured out-of-domain threshold.
# IMPORTANT:
# This applies to the BGE cosine similarity,
# NOT the cross-encoder score.
RELEVANCE_THRESHOLD = 0.60


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


# ============================================================
# LOAD MODELS
# ============================================================

print(
    f"Loading embedding model: "
    f"{EMBEDDING_MODEL_NAME}"
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print(
    f"Loading reranker: "
    f"{RERANKER_MODEL_NAME}"
)

reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)


# ============================================================
# LOAD EMBEDDINGS + METADATA
# ============================================================

chunk_embeddings = np.load(
    EMBEDDINGS_PATH
)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(file)


if len(chunk_embeddings) != len(metadata):
    raise ValueError(
        "Number of embeddings does not match "
        "number of metadata entries."
    )


# ============================================================
# HELPERS
# ============================================================

def get_chunk_id(chunk):

    if "local_chunk_id" in chunk:
        return chunk["local_chunk_id"]

    return chunk["chunk_id"]


# ============================================================
# STAGE 1: DENSE RETRIEVAL
# ============================================================

def dense_retrieve(
    question,
    initial_k=INITIAL_K,
):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Embeddings are normalized, so dot product = cosine.
    scores = (
        chunk_embeddings
        @ query_embedding
    )

    ranking = np.argsort(
        scores
    )[::-1]

    ranking = ranking[
        :min(
            initial_k,
            len(ranking),
        )
    ]

    results = []

    for index in ranking:

        chunk = metadata[index]

        results.append(
            {
                "source": chunk["source"],
                "chunk_id": get_chunk_id(chunk),
                "token_count": chunk.get(
                    "token_count",
                    None,
                ),
                "text": chunk["text"],
                "dense_score": float(
                    scores[index]
                ),
            }
        )

    return results


# ============================================================
# STAGE 2: CROSS-ENCODER RERANKING
# ============================================================

def rerank(
    question,
    candidates,
):

    if not candidates:
        return []

    pairs = [
        (
            question,
            candidate["text"],
        )
        for candidate in candidates
    ]

    reranker_scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = []

    for candidate, score in zip(
        candidates,
        reranker_scores,
    ):

        result = candidate.copy()

        result["reranker_score"] = float(
            score
        )

        reranked.append(result)

    reranked.sort(
        key=lambda x:
        x["reranker_score"],
        reverse=True,
    )

    return reranked


# ============================================================
# COMPLETE RETRIEVAL PIPELINE
# ============================================================

def retrieve(
    question,
    final_k=FINAL_K,
):

    # --------------------------------------------------------
    # Stage 1: dense retrieval
    # --------------------------------------------------------

    dense_candidates = dense_retrieve(
        question,
        INITIAL_K,
    )

    if not dense_candidates:
        return []

    # --------------------------------------------------------
    # Out-of-domain rejection
    # --------------------------------------------------------

    top_dense_score = dense_candidates[
        0
    ]["dense_score"]

    if (
        top_dense_score
        < RELEVANCE_THRESHOLD
    ):

        return []

    # --------------------------------------------------------
    # Stage 2: reranking
    # --------------------------------------------------------

    reranked = rerank(
        question,
        dense_candidates,
    )

    return reranked[:final_k]


# ============================================================
# COMMAND-LINE QUERY LOOP
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SEMANTIC RAG RETRIEVAL")
    print("=" * 70)

    print(
        f"Dense candidates : {INITIAL_K}"
    )

    print(
        f"Final retrieved  : {FINAL_K}"
    )

    print(
        f"Relevance cutoff : "
        f"{RELEVANCE_THRESHOLD}"
    )

    print()

    while True:

        question = input(
            "Question "
            "(or 'exit'): "
        ).strip()

        if (
            question.lower()
            in {
                "exit",
                "quit",
            }
        ):
            break

        if not question:
            continue

        results = retrieve(
            question
        )

        print()

        if not results:

            print(
                "No relevant context found."
            )

            print()
            continue

        print(
            f"Top {len(results)} "
            f"reranked chunks:"
        )

        print()

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                "-" * 70
            )

            print(
                f"Rank: {rank}"
            )

            print(
                f"Source: "
                f"{result['source']}"
            )

            print(
                f"Chunk: "
                f"{result['chunk_id']}"
            )

            print(
                f"Dense score: "
                f"{result['dense_score']:.4f}"
            )

            print(
                f"Reranker score: "
                f"{result['reranker_score']:.4f}"
            )

            print()

            print(
                result["text"]
            )

            print()

        print(
            "=" * 70
        )

        print()


if __name__ == "__main__":
    main()