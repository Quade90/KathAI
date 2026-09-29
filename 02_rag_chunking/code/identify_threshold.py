from pathlib import Path
import csv
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EVAL_DIR = BASE_DIR / "evaluation"

EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR / "semantic_chunk_embeddings.npy"
)

METADATA_PATH = (
    EMBEDDINGS_DIR / "semantic_chunk_metadata.json"
)

OUTPUT_CSV = (
    EVAL_DIR / "threshold_test_results.csv"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = {

    "in_domain": [

        # ---------------- Veritasium / Zippers ----------------

        "Who developed the design that became the modern zipper?",

        "Why do so many zippers have YKK written on them?",

        "How does the slider make the two sides of a zipper interlock?",

        "Why was Whitcomb Judson's early fastener commercially unsuccessful?",

        "What advantage does a coil zipper have over a traditional toothed zipper?",

        "How does the locking mechanism prevent a zipper from opening accidentally?",


        # ---------------- Neo / Presidential Motorcade ----------------

        "What role does the route car play in the presidential motorcade?",

        "How does barrage jamming protect the motorcade from remotely triggered explosives?",

        "What is the purpose of the counter-assault team?",

        "Why are there two identical presidential limousines?",

        "What security features does the Beast have?",

        "How do the vehicles surrounding the president create a protective buffer?",


        # ---------------- 3Blue1Brown / Visual Proofs ----------------

        "Why does the visual argument for the sphere's surface area give the wrong result?",

        "Why does the pi equals four argument fail even though the jagged curves approach a circle?",

        "What are the three false proofs discussed in the video?",

        "What mistake is made in the proof that all triangles are isosceles?",

        "Why can't the curved wedges of a sphere simply be flattened like slices of a circle?",

        "What lesson does the video give about relying on visual intuition in mathematics?"
    ],


    "borderline": [

        "When was YKK founded?",

        "What materials are modern waterproof zippers made from?",

        "How fast can the presidential limousine travel?",

        "How many Secret Service agents travel in a typical motorcade?",

        "What is the formal mathematical definition of uniform convergence?",

        "Who first proved the correct formula for the surface area of a sphere?"
    ],


    "unrelated": [

        "What is McQueen's first name?",

        "How does photosynthesis work?",

        "What is the capital of Argentina?",

        "Why do black holes emit Hawking radiation?",

        "How does TCP congestion control work?",

        "What caused the French Revolution?",

        "How do lithium-ion batteries store energy?",

        "What is the tallest mountain on Mars?"
    ]
}


# ============================================================
# LOAD MODEL + DATA
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


print("Loading chunk embeddings...")

chunk_embeddings = np.load(EMBEDDINGS_PATH)


with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


print(f"Loaded {len(metadata)} chunks.\n")


# ============================================================
# RETRIEVAL
# ============================================================

def get_top_result(question):
    """
    Embed a question and return the highest-scoring chunk.
    """

    question_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = (
        chunk_embeddings @ question_embedding
    )

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    best_chunk = metadata[best_index]

    return {
        "score": best_score,
        "source": best_chunk["source"],
        "chunk_id": best_chunk["local_chunk_id"],
        "text": best_chunk["text"]
    }


# ============================================================
# RUN TEST
# ============================================================

def main():

    results = []

    print("=" * 80)
    print("THRESHOLD TEST")
    print("=" * 80)

    for category, questions in TEST_QUESTIONS.items():

        print(f"\nCATEGORY: {category.upper()}")
        print("-" * 80)

        for question in questions:

            result = get_top_result(question)

            row = {
                "category": category,
                "question": question,
                "top_score": result["score"],
                "source": result["source"],
                "chunk_id": result["chunk_id"]
            }

            results.append(row)

            print(
                f"\nQuestion: {question}"
                f"\nScore:    {result['score']:.4f}"
                f"\nSource:   {result['source']}"
                f"\nChunk:    {result['chunk_id']}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "question",
                "top_score",
                "source",
                "chunk_id"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for category in TEST_QUESTIONS.keys():

        scores = [
            r["top_score"]
            for r in results
            if r["category"] == category
        ]

        scores = np.array(scores)

        print(
            f"\n{category.upper()}"
            f"\nCount:  {len(scores)}"
            f"\nMean:   {scores.mean():.4f}"
            f"\nMin:    {scores.min():.4f}"
            f"\nMax:    {scores.max():.4f}"
            f"\nMedian: {np.median(scores):.4f}"
        )

    print(
        f"\nResults saved to:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()