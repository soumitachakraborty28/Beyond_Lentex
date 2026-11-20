# Beyond LentEx: Multi-Entity Extraction and Adaptive Entity-Filtered RAG

## Overview
Beyond LentEx explores extensions to entity-aware Retrieval-Augmented Generation (RAG) through multi-entity selection, semantic ranking, and adaptive entity selection.

The project investigates whether considering multiple query entities can improve document retrieval compared with a single-entity approach.

## Key Components

- Candidate entity generation using lexical extraction.
- Semantic representation using SentenceTransformers.
- Entity ranking using cosine similarity.
- Single-entity and multi-entity retrieval (Top-1, Top-3, Top-5).
- Adaptive-K retrieval strategy.
- Retrieval evaluation using Precision, Recall and F1-score.
- Bootstrap-based comparison of Adaptive-K against the Top-1 baseline.

## Methodology

Query → Candidate Generation → Embedding Generation → Similarity Ranking → Entity Filtering → Document Retrieval → Evaluation.

## Technologies Used

- Python
- SentenceTransformers
- Scikit-learn
- NumPy
- Pandas
- Cosine Similarity

## Experimental Results

The current experimental run produced:

| Method | F1-score |
|---|---:|
| Vanilla RAG | 0.743 |
| Multi-Entity Top-1 | 0.000 |
| Multi-Entity Top-3 | 0.333 |
| Multi-Entity Top-5 | 0.733 |
| Adaptive-K | 0.733 |

## Dataset

The current implementation uses a local experiment.json file containing queries, documents and relevance information.

This is the experimental dataset used for the current run, not a reproduction of the official COLIEE or BioASQ benchmarks.

## Important Experimental Limitations

- Candidate entities are generated using lightweight lexical extraction rather than a dedicated Named Entity Recognition model.
- Adaptive-K selected K=5 consistently in the current run.
- Published LentEx values are reference results, not results reproduced by this implementation.
- The current experiment does not establish a direct performance improvement over the original LentEx model.
- Results should be interpreted within the scope of the current experimental dataset.

## Future Scope

- Evaluate on official benchmark datasets.
- Integrate a dedicated entity extraction model.
- Improve adaptive threshold selection.
- Test across larger datasets.
- Perform direct comparisons with the original LentEx implementation.
