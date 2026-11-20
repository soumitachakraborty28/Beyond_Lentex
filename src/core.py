import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PUBLISHED LENTEX REFERENCE VALUES
# These are literature values, NOT measured by this experiment.
# ============================================================

PUBLISHED = {
    "COLIEE": {
        "Vanilla RAG": {
            "precision": 0.248,
            "recall": 0.167,
            "f1": 0.199
        },
        "RAG + LentEx": {
            "precision": 0.792,
            "recall": 0.681,
            "f1": 0.732
        }
    },

    "BioASQ": {
        "Vanilla RAG": {
            "precision": 0.612,
            "recall": 0.441,
            "f1": 0.513
        },
        "RAG + LentEx": {
            "precision": 0.742,
            "recall": 0.595,
            "f1": 0.660
        }
    }
}


# ============================================================
# BASIC UTILITIES
# ============================================================

def norm(text):
    """
    Normalize text for exact entity/document matching.
    """
    text = re.sub(r"[^\w\s-]", " ", str(text).lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prf(retrieved, gold):
    """
    Precision, Recall and F1.
    """
    retrieved_set = set(norm(x) for x in retrieved)
    gold_set = set(norm(x) for x in gold)

    tp = len(retrieved_set & gold_set)

    precision = (
        tp / len(retrieved_set)
        if retrieved_set
        else 0.0
    )

    recall = (
        tp / len(gold_set)
        if gold_set
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


def precision_at_k(ranked_items, gold_items, k):
    """
    Precision@K for ranked entities/documents.
    """
    ranked = [norm(x) for x in ranked_items[:k]]
    gold = set(norm(x) for x in gold_items)

    if not ranked:
        return 0.0

    return len(set(ranked) & gold) / len(ranked)


def recall_at_k(ranked_items, gold_items, k):
    """
    Recall@K for ranked entities/documents.
    """
    ranked = [norm(x) for x in ranked_items[:k]]
    gold = set(norm(x) for x in gold_items)

    if not gold:
        return 0.0

    return len(set(ranked) & gold) / len(gold)


def f1_at_k(ranked_items, gold_items, k):
    """
    F1@K.
    """
    p = precision_at_k(ranked_items, gold_items, k)
    r = recall_at_k(ranked_items, gold_items, k)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def reciprocal_rank(ranked_items, gold_items):
    """
    Reciprocal Rank / MRR component.

    If the first relevant entity appears at position 1:
        RR = 1

    position 2:
        RR = 0.5

    etc.
    """
    gold = set(norm(x) for x in gold_items)

    for position, item in enumerate(ranked_items, start=1):
        if norm(item) in gold:
            return 1.0 / position

    return 0.0


# ============================================================
# ADAPTIVE K
# ============================================================
def adaptive_k(scores, min_k=1, max_k=5, threshold=0.60, gap=0.15):
    if not scores:
        return 0

    n = min(len(scores), max_k)

    if n <= min_k:
        return n

    k = min_k

    for i in range(1, n):
        previous_score = scores[i - 1]
        current_score = scores[i]

        relative_gap = (
            max(0, previous_score - current_score)
            / max(abs(previous_score), 1e-8)
        )

        if current_score >= threshold and relative_gap <= gap:
            k = i + 1
        else:
            break

    return min(k, n)

# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

def bootstrap(a, b, B=5000, seed=42):

    rng = np.random.default_rng(seed)

    a = np.asarray(a)
    b = np.asarray(b)

    differences = a - b

    if len(differences) == 0:
        return {
            "difference": 0,
            "ci_low": 0,
            "ci_high": 0,
            "p": 1
        }

    simulated_means = []

    for _ in range(B):

        sample_indices = rng.integers(
            0,
            len(differences),
            len(differences)
        )

        sample = differences[sample_indices]

        simulated_means.append(
            sample.mean()
        )

    simulated_means = np.asarray(simulated_means)

    observed_difference = differences.mean()

    ci_low = np.percentile(
        simulated_means,
        2.5
    )

    ci_high = np.percentile(
        simulated_means,
        97.5
    )

    p_value = min(
        1.0,
        2 * min(
            np.mean(simulated_means <= 0),
            np.mean(simulated_means >= 0)
        )
    )

    return {
        "difference": float(observed_difference),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p": float(p_value)
    }


# ============================================================
# EXPERIMENT
# ============================================================

class Experiment:

    def __init__(self, rows, model_name):

        self.rows = rows

        print(
            f"\nLoading embedding model: {model_name}\n"
        )

        self.model = SentenceTransformer(
            model_name
        )


    # ========================================================
    # MULTI-ENTITY CANDIDATE GENERATION
    # ========================================================

    def entities_from_text(self, query):

        words = re.findall(
            r"[A-Za-z][A-Za-z-]{2,}",
            query.lower()
        )

        stop_words = {
            "what",
            "which",
            "where",
            "when",
            "does",
            "with",
            "from",
            "into",
            "about",
            "that",
            "this",
            "these",
            "those",
            "have",
            "has",
            "been",
            "were",
            "their",
            "they",
            "there",
            "should",
            "could",
            "would",
            "between",
            "under",
            "over",
            "and",
            "the",
            "for",
            "are",
            "was",
            "can"
        }

        words = [
            word
            for word in words
            if word not in stop_words
        ]

        candidates = []

        # ----------------------------------------------------
        # Generate 1-word, 2-word and 3-word candidates.
        # ----------------------------------------------------

        for i in range(len(words)):

            # Unigram
            candidates.append(
                words[i]
            )

            # Bigram
            if i + 1 < len(words):

                candidates.append(
                    words[i]
                    + " "
                    + words[i + 1]
                )

            # Trigram
            if i + 2 < len(words):

                candidates.append(
                    words[i]
                    + " "
                    + words[i + 1]
                    + " "
                    + words[i + 2]
                )

        # ----------------------------------------------------
        # Remove duplicates while preserving extraction order.
        # ----------------------------------------------------

        seen = set()
        result = []

        for candidate in candidates:

            candidate = norm(candidate)

            if (
                candidate
                and candidate not in seen
            ):

                seen.add(candidate)
                result.append(candidate)

        return result[:20]


    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(self, outdir):

        outdir = Path(outdir)

        outdir.mkdir(
            parents=True,
            exist_ok=True
        )

        per_query_results = []

        entity_rows = []

        ranking_rows = []

        ranking_details = []

        # ====================================================
        # QUERY LOOP
        # ====================================================

        for row in self.rows:

            query_id = row["query_id"]

            query = row["query"]

            print(
                f"\nProcessing query: {query_id}"
            )

            # ------------------------------------------------
            # STEP 1 — Candidate extraction
            # ------------------------------------------------

            candidates = self.entities_from_text(
                query
            )

            if not candidates:
                continue

            # ------------------------------------------------
            # STEP 2 — Query embedding
            # ------------------------------------------------

            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True
            )[0]

            # ------------------------------------------------
            # STEP 3 — Entity embeddings
            # ------------------------------------------------

            entity_embeddings = self.model.encode(
                candidates,
                normalize_embeddings=True
            )

            # ------------------------------------------------
            # STEP 4 — Semantic similarity
            # ------------------------------------------------

            scores = (
                entity_embeddings
                @ query_embedding
            ).tolist()

            # ------------------------------------------------
            # EXTRACTION ORDER
            # ------------------------------------------------

            extraction_order = candidates.copy()

            # ------------------------------------------------
            # SEMANTIC RANKING
            # ------------------------------------------------

            ranked_pairs = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )

            similarity_rank = [
                entity
                for entity, score
                in ranked_pairs
            ]

            similarity_scores = [
                float(score)
                for entity, score
                in ranked_pairs
            ]

            # ------------------------------------------------
            # Save ranking evidence
            # ------------------------------------------------

            for position, (
                entity,
                score
            ) in enumerate(
                ranked_pairs,
                start=1
            ):

                ranking_details.append({
                    "query_id": query_id,
                    "entity": entity,
                    "similarity_score": score,
                    "rank": position
                })

            # =================================================
            # GOLD ENTITIES
            # =================================================

            gold_entities = [
                norm(entity)
                for entity
                in row.get(
                    "gold_entities",
                    []
                )
            ]

            # =================================================
            # RESEARCH GAP 1
            # SINGLE vs MULTI ENTITY
            # =================================================

            if gold_entities:

                # Single entity
                p1, r1, f1 = prf(
                    extraction_order[:1],
                    gold_entities
                )

                # Multi entity
                pm, rm, fm = prf(
                    similarity_rank[:5],
                    gold_entities
                )

                entity_rows.append({
                    "query_id": query_id,
                    "method": "Single/Top-1",
                    "precision": p1,
                    "recall": r1,
                    "f1": f1,
                    "count": 1
                })

                entity_rows.append({
                    "query_id": query_id,
                    "method": "Multi-Entity Top-5",
                    "precision": pm,
                    "recall": rm,
                    "f1": fm,
                    "count": min(
                        5,
                        len(similarity_rank)
                    )
                })

            # =================================================
            # RESEARCH GAP 1B
            # EXTRACTION ORDER vs SEMANTIC RANKING
            # =================================================

            if gold_entities:

                for method_name, ranked_entities in [
                    (
                        "Extraction Order",
                        extraction_order
                    ),
                    (
                        "Similarity Ranking",
                        similarity_rank
                    )
                ]:

                    for k in [1, 3, 5]:

                        p_at_k = precision_at_k(
                            ranked_entities,
                            gold_entities,
                            k
                        )

                        r_at_k = recall_at_k(
                            ranked_entities,
                            gold_entities,
                            k
                        )

                        f_at_k = f1_at_k(
                            ranked_entities,
                            gold_entities,
                            k
                        )

                        ranking_rows.append({

                            "query_id": query_id,

                            "ranking": method_name,

                            "K": k,

                            "precision_at_k": p_at_k,

                            "recall_at_k": r_at_k,

                            "f1_at_k": f_at_k,

                            "MRR": reciprocal_rank(
                                ranked_entities,
                                gold_entities
                            )
                        })

            # =================================================
            # DOCUMENT RETRIEVAL
            # =================================================

            documents = row["documents"]

            gold_documents = [
                document["doc_id"]
                for document in documents
                if document.get(
                    "relevant",
                    False
                )
            ]

            # ------------------------------------------------
            # Document embeddings
            # ------------------------------------------------

            document_embeddings = self.model.encode(
                [
                    document["text"]
                    for document in documents
                ],
                normalize_embeddings=True
            )

            # ------------------------------------------------
            # Document similarity
            # ------------------------------------------------

            document_scores = (
                document_embeddings
                @ query_embedding
            ).tolist()

            document_order = np.argsort(
                document_scores
            )[::-1].tolist()

            # =================================================
            # FIXED K METHODS
            # =================================================

            methods = {

                "Vanilla RAG": None,

                "Multi-Entity Top-1": 1,

                "Multi-Entity Top-3": 3,

                "Multi-Entity Top-5": 5
            }

            # =================================================
            # NORMALIZE ENTITY SCORES
            # =================================================

            score_array = np.asarray(
                similarity_scores
            )

            if len(score_array):

                minimum = score_array.min()

                maximum = score_array.max()

                normalized_scores = (
                    (
                        score_array
                        - minimum
                    )
                    /
                    (
                        maximum
                        - minimum
                        + 1e-9
                    )
                ).tolist()

            else:

                normalized_scores = []

            # =================================================
            # ADAPTIVE K
            # =================================================

            selected_k = adaptive_k(
                normalized_scores,
                min_k=1,
                max_k=5,
                threshold=0.50,
                gap=0.25
            )

            methods[
                "Adaptive-K"
            ] = selected_k

            # =================================================
            # RETRIEVAL FOR EACH METHOD
            # =================================================

            for method_name, k in methods.items():

                # ------------------------------------------------
                # Vanilla RAG
                # ------------------------------------------------

                if k is None:

                    candidate_document_indices = (
                        document_order
                    )

                # ------------------------------------------------
                # Entity-filtered retrieval
                # ------------------------------------------------

                else:

                    selected_entities = [
                        norm(entity)
                        for entity
                        in similarity_rank[:k]
                    ]

                    filtered_indices = []

                    for index, document in enumerate(
                        documents
                    ):

                        document_text = norm(
                            document["text"]
                        )

                        # A document passes the filter
                        # if it contains at least one
                        # selected entity.

                        if any(
                            entity in document_text
                            for entity
                            in selected_entities
                        ):

                            filtered_indices.append(
                                index
                            )

                    # Semantic reranking
                    candidate_document_indices = sorted(
                        filtered_indices,
                        key=lambda index:
                        document_scores[index],
                        reverse=True
                    )

                # ------------------------------------------------
                # Top-10 retrieved documents
                # ------------------------------------------------

                retrieved_documents = [
                    documents[index]["doc_id"]
                    for index
                    in candidate_document_indices[:10]
                ]

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                precision, recall, f1 = prf(
                    retrieved_documents,
                    gold_documents
                )

                per_query_results.append({

                    "query_id": query_id,

                    "method": method_name,

                    "precision": precision,

                    "recall": recall,

                    "f1": f1,

                    "selected_K": k
                })

        # ====================================================
        # DATAFRAMES
        # ====================================================

        per_query_df = pd.DataFrame(
            per_query_results
        )

        entity_df = pd.DataFrame(
            entity_rows
        )

        ranking_df = pd.DataFrame(
            ranking_rows
        )

        ranking_details_df = pd.DataFrame(
            ranking_details
        )

        # ====================================================
        # RETRIEVAL SUMMARY
        # ====================================================

        retrieval_summary = (
            per_query_df
            .groupby("method")
            .agg(

                precision=(
                    "precision",
                    "mean"
                ),

                recall=(
                    "recall",
                    "mean"
                ),

                f1=(
                    "f1",
                    "mean"
                ),

                avg_k=(
                    "selected_K",
                    "mean"
                ),

                median_k=(
                    "selected_K",
                    "median"
                ),

                min_k=(
                    "selected_K",
                    "min"
                ),

                max_k=(
                    "selected_K",
                    "max"
                )
            )
            .reset_index()
        )

        # ====================================================
        # ADAPTIVE-K vs TOP-1
        # ====================================================

        wide = (
            per_query_df
            .pivot(
                index="query_id",
                columns="method",
                values="f1"
            )
        )

        required_columns = [
            "Adaptive-K",
            "Multi-Entity Top-1"
        ]

        available_columns = [
            column
            for column
            in required_columns
            if column in wide.columns
        ]

        if len(available_columns) == 2:

            comparison = (
                wide[
                    required_columns
                ]
                .dropna()
            )

            bootstrap_result = bootstrap(
                comparison["Adaptive-K"].values,
                comparison[
                    "Multi-Entity Top-1"
                ].values
            )

        else:

            bootstrap_result = {
                "difference": 0,
                "ci_low": 0,
                "ci_high": 0,
                "p": 1
            }

        # ====================================================
        # SAVE CSV FILES
        # ====================================================

        per_query_df.to_csv(
            outdir /
            "per_query_results.csv",
            index=False
        )

        retrieval_summary.to_csv(
            outdir /
            "retrieval_summary.csv",
            index=False
        )

        entity_df.to_csv(
            outdir /
            "entity_gap_results.csv",
            index=False
        )

        ranking_df.to_csv(
            outdir /
            "ranking_gap_results.csv",
            index=False
        )

        ranking_details_df.to_csv(
            outdir /
            "ranking_details.csv",
            index=False
        )

        (
            outdir /
            "adaptive_vs_top1_bootstrap.json"
        ).write_text(
            json.dumps(
                bootstrap_result,
                indent=2
            ),
            encoding="utf-8"
        )

        # ====================================================
        # REPORT TABLE
        # ====================================================

        report_path = (
            outdir /
            "REPORT_TABLES.md"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as report:

            report.write(
                "# Beyond LentEx — Experiment Tables\n\n"
            )

            # ------------------------------------------------
            # TABLE A
            # ------------------------------------------------

            report.write(
                "## A. Retrieval results measured by this run\n\n"
            )

            report.write(
                retrieval_summary.to_markdown(
                    index=False,
                    floatfmt=".3f"
                )
            )

            # ------------------------------------------------
            # TABLE B
            # ------------------------------------------------

            report.write(
                "\n\n"
                "## B. Entity-level research-gap results\n\n"
            )

            if len(entity_df):

                entity_summary = (
                    entity_df
                    .groupby("method")
                    [
                        [
                            "precision",
                            "recall",
                            "f1",
                            "count"
                        ]
                    ]
                    .mean()
                    .reset_index()
                )

                report.write(
                    entity_summary.to_markdown(
                        index=False,
                        floatfmt=".3f"
                    )
                )

            else:

                report.write(
                    "No gold entity labels supplied.\n"
                )

            # ------------------------------------------------
            # TABLE C
            # ------------------------------------------------

            report.write(
                "\n\n"
                "## C. Ranking research-gap results\n\n"
            )

            if len(ranking_df):

                ranking_summary = (
                    ranking_df
                    .groupby(
                        [
                            "ranking",
                            "K"
                        ]
                    )
                    [
                        [
                            "precision_at_k",
                            "recall_at_k",
                            "f1_at_k",
                            "MRR"
                        ]
                    ]
                    .mean()
                    .reset_index()
                )

                report.write(
                    ranking_summary.to_markdown(
                        index=False,
                        floatfmt=".3f"
                    )
                )

            else:

                report.write(
                    "No gold entity labels supplied.\n"
                )

            # ------------------------------------------------
            # TABLE D
            # ------------------------------------------------

            report.write(
                "\n\n"
                "## D. Adaptive-K vs Top-1 paired bootstrap\n\n"
            )

            report.write(
                json.dumps(
                    bootstrap_result,
                    indent=2
                )
            )

            # ------------------------------------------------
            # TABLE E
            # ------------------------------------------------

            report.write(
                "\n\n"
                "## E. Published LentEx reference values "
                "(NOT measured by this run)\n\n"
            )

            report.write(
                "| Dataset | Vanilla RAG F1 | LentEx F1 |\n"
            )

            report.write(
                "|---|---:|---:|\n"
            )

            for dataset, values in PUBLISHED.items():

                report.write(
                    f"| {dataset} | "
                    f"{values['Vanilla RAG']['f1']:.3f} | "
                    f"{values['RAG + LentEx']['f1']:.3f} |\n"
                )

        print(
            "\n=========================================="
        )

        print(
            "EXPERIMENT COMPLETED"
        )

        print(
            "==========================================\n"
        )

        print(
            retrieval_summary.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.3f}"
            )
        )

        print(
            "\nAdaptive-K vs Top-1:"
        )

        print(
            json.dumps(
                bootstrap_result,
                indent=2
            )
        )

        print(
            f"\nFull report saved to:\n"
            f"{report_path.resolve()}"
        )

        return (
            retrieval_summary,
            bootstrap_result
        )