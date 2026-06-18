# Vocabulary mismatch: grep vs semantic vs LLM search

**Question.** Vocabulary mismatch is the classic weakness of lexical search: the query and the
answer use different words. How much does it actually cost grep/BM25, how well does semantic
(embedding) search close the gap, and where do LLMs sit — depending on *how* the LLM is used
(writing the query blind, vs. seeing the documents and query together)?

**Setup.** Corpus = 8,941 human-written passages from MS MARCO v2.1; queries = real typed Bing
searches; each query's `is_selected` passage is its gold answer. Queries are bucketed by **stemmed
term overlap** between the query and its gold passage — the vocabulary-mismatch axis:

- `zero` — no query word-stem appears in the answer (genuine mismatch). 104 queries (12%).
- `low`  — <50% of query stems appear. 200 (22%).
- `high` — >=50% appear. 600 (66%).

Stemming matters: without it, plurals (`psychopath`/`psychopaths`) masquerade as mismatch and
make grep look artificially bad. All lexical arms are stemmed so the gap measured is *real*
vocabulary mismatch (synonyms, paraphrase, acronyms), not morphology.

## Result 1 — full corpus, retrieval-only arms (n=904, 9k docs)

| arm | regime | hit@1 | hit@10 | MRR@10 | **zero** h@10 | low h@10 | high h@10 | lat/query | index |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| grep (naive ripgrep) | lexical | 0.182 | 0.614 | 0.315 | **0.00** | 0.23 | 0.85 | 6.17 ms | 0.0 s |
| BM25 (stemmed) | lexical | 0.181 | 0.755 | 0.345 | **0.00** | 0.57 | 0.95 | 2.23 ms | 5.69 s |
| potion-32M (static emb) | semantic | 0.224 | 0.886 | 0.413 | **0.56** | 0.90 | 0.94 | 0.02 ms | 0.38 s |
| MiniLM (dense emb) | semantic | 0.363 | 0.954 | 0.561 | **0.82** | 0.93 | 0.99 | 1.34 ms | 11.0 s |

In the `zero` bucket, **grep and BM25 score exactly 0.00 hit@10** — a hard floor, by construction:
they cannot match words that are not in the document. Dense MiniLM recovers
**82%** of those same queries, and is *faster per query*
(1.3 ms) than BM25 (2.2 ms) once indexed.

## Result 2 — all arms incl. LLM, same 90-query subset (35 zero / 35 low / 20 high)

| arm | regime | hit@1 | hit@10 | MRR@10 | **zero** h@10 | low h@10 | high h@10 |
|---|---|--:|--:|--:|--:|--:|--:|
| grep (naive ripgrep) | lexical | 0.067 | 0.311 | 0.143 | **0.00** | 0.26 | 0.95 |
| bm25 (lexical, stemmed) | lexical | 0.078 | 0.433 | 0.166 | **0.00** | 0.54 | 1.00 |
| bm25 + LLM-expand [2a] | LLM writes query (blind) | 0.122 | 0.700 | 0.294 | **0.54** | 0.69 | 1.00 |
| claude-grep agentic [2b] | LLM writes query (agentic) | 0.378 | 0.878 | 0.540 | **0.77** | 0.94 | 0.95 |
| potion (semantic static) | embeddings | 0.122 | 0.789 | 0.302 | **0.49** | 0.97 | 1.00 |
| minilm (semantic dense) | embeddings | 0.300 | 0.878 | 0.480 | **0.83** | 0.86 | 1.00 |
| LLM-rerank / lexical cands [4a-lex] | LLM reads docs+query | 0.256 | 0.478 | 0.323 | **0.00** | 0.69 | 0.95 |
| LLM-rerank / semantic cands [4a-sem] | LLM reads docs+query | 0.333 | 0.922 | 0.502 | **0.86** | 0.94 | 1.00 |

## Result 3 — first-stage recall@20 (what reranking can even see)

An LLM reranker can only re-order what the first-stage retriever handed it. % of golds present
in the top-20 candidate pool:

| first stage | zero | low | high | all |
|---|--:|--:|--:|--:|
| BM25 (lexical)   | **0.00** | 0.69 | 1.00 | 0.49 |
| MiniLM (semantic)| **0.86** | 0.94 | 1.00 | 0.92 |

This is why `rerank/lexical` scores 0.00 on the zero bucket — identical to BM25. The gold is
never in the pool, so the LLM cannot recover it no matter how well it reads.

## Result 4 — latency & cost

| class | per-query latency | per-query cost | one-time |
|---|---|---|---|
| grep | ~6 ms | free | none |
| BM25 | ~2 ms | free | 5.7 s index |
| potion (static) | ~0.02 ms | free | 0.4 s index |
| MiniLM (dense) | ~1.3 ms | free | 11 s index |
| LLM arms | seconds–minutes | ~9,516 tokens (avg) | — |

The LLM arms together: 123 agents, 3,416,164 tokens, 824 tool
calls, 992 s wall-clock at ~16x concurrency. Agentic grep dominates (many
iterative searches); rerank is mid; blind expansion is cheap. That is **~10,000x the per-query**
**latency of dense retrieval** for, at best, a few points of hit@10.

## Findings

1. **Vocabulary mismatch is a hard zero for lexical search, not a soft penalty.** grep and BM25
   get 0.00 hit@10 on genuine zero-overlap queries. The effect is real but *rare in a natural query
   log* — only ~12% of MS MARCO queries are true zero-overlap — yet that slice is exactly where
   lexical is blind.

2. **Your distinction holds and has a sharp boundary.** 'LLM sees documents + query together'
   (rerank over semantic candidates: 0.86 on zero) beats 'LLM writes the query' — but the latter
   splits by agency: blind one-shot expansion gets 0.54, while *agentic* grep (iterate, read hits,
   guess synonyms) reaches 0.77. The more the LLM can see partial results and adapt, the more of
   the gap it closes.

3. **The gap-closing power is in the retrieval, not the reading.** LLM rerank over *lexical*
   candidates = 0.00 on zero (no better than grep); over *semantic* candidates = 0.86. Letting an
   LLM read the documents does nothing for vocabulary mismatch if those documents were fetched
   lexically. Rerank inherits the first stage's blindness.

4. **Plain dense embeddings are the value winner.** MiniLM matches agentic-grep on overall hit@10
   (0.88), beats it on the zero bucket (0.83 vs 0.77), at ~1 ms and zero token cost per query vs
   seconds and thousands of tokens. The only arm that clearly beats it (rerank-sem, 0.92) *uses
   MiniLM as its first stage* and pays ~10,000x latency for +4 points.

**Bottom line.** For non-code text where queries are human-phrased, a small dense-embedding index
closes most of the vocabulary-mismatch gap cheaply. An LLM helps further only when it reads
semantically-retrieved candidates; an LLM that merely writes grep queries helps less, and an LLM
reranking lexical results inherits grep's blind spot entirely.