# Beyond LentEx — Experiment Tables

## A. Retrieval results measured by this run

| method             |   precision |   recall |    f1 |   avg_k |   median_k |   min_k |   max_k |
|:-------------------|------------:|---------:|------:|--------:|-----------:|--------:|--------:|
| Adaptive-K         |       0.800 |    0.700 | 0.733 |   5.000 |      5.000 |   5.000 |   5.000 |
| Multi-Entity Top-1 |       0.000 |    0.000 | 0.000 |   1.000 |      1.000 |   1.000 |   1.000 |
| Multi-Entity Top-3 |       0.400 |    0.300 | 0.333 |   3.000 |      3.000 |   3.000 |   3.000 |
| Multi-Entity Top-5 |       0.800 |    0.700 | 0.733 |   5.000 |      5.000 |   5.000 |   5.000 |
| Vanilla RAG        |       0.600 |    1.000 | 0.743 | nan     |    nan     | nan     | nan     |

## B. Entity-level research-gap results

| method             |   precision |   recall |    f1 |   count |
|:-------------------|------------:|---------:|------:|--------:|
| Multi-Entity Top-5 |       0.160 |    0.333 | 0.214 |   5.000 |
| Single/Top-1       |       0.000 |    0.000 | 0.000 |   1.000 |

## C. Ranking research-gap results

| ranking            |   K |   precision_at_k |   recall_at_k |   f1_at_k |   MRR |
|:-------------------|----:|-----------------:|--------------:|----------:|------:|
| Extraction Order   |   1 |            0.000 |         0.000 |     0.000 | 0.500 |
| Extraction Order   |   3 |            0.333 |         0.433 |     0.373 | 0.500 |
| Extraction Order   |   5 |            0.200 |         0.433 |     0.271 | 0.500 |
| Similarity Ranking |   1 |            0.000 |         0.000 |     0.000 | 0.247 |
| Similarity Ranking |   3 |            0.133 |         0.167 |     0.147 | 0.247 |
| Similarity Ranking |   5 |            0.160 |         0.333 |     0.214 | 0.247 |

## D. Adaptive-K vs Top-1 paired bootstrap

{
  "difference": 0.7333333333333333,
  "ci_low": 0.3333333333333333,
  "ci_high": 1.0,
  "p": 0.0012
}

## E. Published LentEx reference values (NOT measured by this run)

| Dataset | Vanilla RAG F1 | LentEx F1 |
|---|---:|---:|
| COLIEE | 0.199 | 0.732 |
| BioASQ | 0.513 | 0.660 |
