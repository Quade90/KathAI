from pathlib import Path
import json
import re
import csv

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EVALUATION_DIR = BASE_DIR / "evaluation"

GROUND_TRUTH_PATH = (
    EVALUATION_DIR / "ground_truth_30.json"
)

OUTPUT_PATH = (
    EVALUATION_DIR / "chunking_comparison.csv"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

K_VALUES = [1, 3, 5]


# ============================================================
# PATHS FOR BOTH METHODS
# ============================================================

SEMANTIC_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR / "semantic_chunk_embeddings.npy"
)

SEMANTIC_METADATA_PATH = (
    EMBEDDINGS_DIR / "semantic_chunk_metadata.json"
)

BASELINE_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR / "baseline_chunk_embeddings.npy"
)

BASELINE_METADATA_PATH = (
    EMBEDDINGS_DIR / "baseline_chunk_metadata.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# LOAD EMBEDDINGS + METADATA
# ============================================================

semantic_embeddings = np.load(
    SEMANTIC_EMBEDDINGS_PATH
)

baseline_embeddings = np.load(
    BASELINE_EMBEDDINGS_PATH
)


with open(
    SEMANTIC_METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    semantic_metadata = json.load(f)


with open(
    BASELINE_METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    baseline_metadata = json.load(f)


with open(
    GROUND_TRUTH_PATH,
    "r",
    encoding="utf-8"
) as f:
    evaluation_questions = json.load(f)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text so harmless punctuation/spacing differences
    do not cause the ground-truth check to fail.
    """

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question,
    embeddings,
    metadata,
    k
):
    """
    Retrieve top-k chunks using cosine similarity.
    Embeddings were normalized earlier, so dot product
    is cosine similarity.
    """

    question_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = (
        embeddings @ question_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:k]

    results = []

    for index in top_indices:

        chunk = metadata[index]

        results.append({
            "source": chunk["source"],
            "chunk_id": chunk["local_chunk_id"],
            "text": chunk["text"],
            "similarity": float(
                similarities[index]
            )
        })

    return results


# ============================================================
# RECONSTRUCT RETRIEVED CONTEXT
# ============================================================

def reconstruct_context(results, source):
    """
    Keep retrieved chunks from the correct source and
    sort them by their original chunk order.

    This allows adjacent fixed chunks to reconstruct
    context that was split across a chunk boundary.
    """

    relevant_chunks = [
        chunk
        for chunk in results
        if chunk["source"] == source
    ]

    relevant_chunks.sort(
        key=lambda x: x["chunk_id"]
    )

    combined_text = " ".join(
        chunk["text"]
        for chunk in relevant_chunks
    )

    return normalize_text(combined_text)


# ============================================================
# GROUND-TRUTH CHECK
# ============================================================

def contains_ground_truth(
    results,
    source,
    ground_truth
):
    """
    Check whether the retrieved chunks collectively contain
    the complete ground-truth passage.
    """

    retrieved_text = reconstruct_context(
        results,
        source
    )

    truth = normalize_text(
        ground_truth
    )

    return truth in retrieved_text


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    rows = []

    semantic_scores = {
        k: []
        for k in K_VALUES
    }

    baseline_scores = {
        k: []
        for k in K_VALUES
    }


    print("\n" + "=" * 80)
    print("SEMANTIC VS FIXED-SIZE CHUNKING")
    print("=" * 80)


    for item in evaluation_questions:

        question = item["question"]
        source = item["source"]
        ground_truth = item["ground_truth"]

        print(f"\nQuestion: {question}")

        for k in K_VALUES:

            # --------------------------------------------
            # Semantic retrieval
            # --------------------------------------------

            semantic_results = retrieve(
                question,
                semantic_embeddings,
                semantic_metadata,
                k
            )

            semantic_success = (
                contains_ground_truth(
                    semantic_results,
                    source,
                    ground_truth
                )
            )


            # --------------------------------------------
            # Baseline retrieval
            # --------------------------------------------

            baseline_results = retrieve(
                question,
                baseline_embeddings,
                baseline_metadata,
                k
            )

            baseline_success = (
                contains_ground_truth(
                    baseline_results,
                    source,
                    ground_truth
                )
            )


            semantic_scores[k].append(
                int(semantic_success)
            )

            baseline_scores[k].append(
                int(baseline_success)
            )


            rows.append({
                "question": question,
                "source": source,
                "k": k,
                "semantic_success":
                    int(semantic_success),
                "baseline_success":
                    int(baseline_success)
            })


            print(
                f"  k={k}: "
                f"Semantic={'✓' if semantic_success else '✗'} | "
                f"Baseline={'✓' if baseline_success else '✗'}"
            )


    # ========================================================
    # SAVE PER-QUESTION RESULTS
    # ========================================================

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "source",
                "k",
                "semantic_success",
                "baseline_success"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


    # ========================================================
    # FINAL RECALL@K
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for k in K_VALUES:

        semantic_recall = np.mean(
            semantic_scores[k]
        )

        baseline_recall = np.mean(
            baseline_scores[k]
        )

        improvement = (
            semantic_recall
            - baseline_recall
        )

        print(
            f"\nRecall@{k}"
            f"\nSemantic: {semantic_recall:.3f}"
            f"\nBaseline: {baseline_recall:.3f}"
            f"\nDifference: {improvement:+.3f}"
        )


    print(
        f"\nDetailed results saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()