# Beyond LentEx — Live Research-Gap Experiment

This runnable experiment is designed around the four research questions in the supplied seminar reports:

G1. Does multi-entity extraction improve entity coverage over a single entity?
G2. Does semantic ranking place useful entities higher than extraction order?
G3. Does fixed Top-K improve retrieval coverage over strict Top-1?
G4. Does Adaptive-K choose a better filter size query-by-query?

The program produces:
1. Entity-level coverage/precision/F1 and redundancy.
2. Ranking comparison: extraction order vs similarity ranking.
3. Retrieval comparison: Vanilla RAG, Top-1, Top-3, Top-5, Adaptive-K.
4. Adaptive-K statistics: average/median/min/max K.
5. Paired bootstrap CI and p-value for Adaptive vs Top-1.
6. CSV/Markdown tables suitable for the seminar report.

IMPORTANT:
- The program does NOT fabricate gold labels.
- If you provide official COLIEE/BioASQ data, it uses them.
- If no official dataset is supplied, `--demo` creates a SMALL SYNTHETIC smoke test so the pipeline can be demonstrated. Demo results MUST NOT be presented as COLIEE/BioASQ results.
- Published LentEx numbers from the paper are stored separately as reference values; they are never mixed with measured values.

For a real run:
  python scripts/run_live.py --dataset path/to/dataset.jsonl --model google/flan-t5-small

Dataset JSONL schema:
{"query_id":"q1","query":"...","documents":[{"doc_id":"d1","text":"...","relevant":true}, ...]}

Optional gold entities:
{"gold_entities":["entity 1","entity 2"]}

The experiment will use gold document relevance for retrieval metrics.
