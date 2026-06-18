"""Render REPORT.md from results/*.json."""
import json

fm = json.load(open("results/final_metrics.json"))
dm = json.load(open("results/det_metrics.json"))
dl = json.load(open("results/det_latency.json"))
use = json.load(open("results/llm_usage.json"))
arms = fm["arms"]; N = fm["subset_n"]
L = []
def p(s=""): L.append(s)

p("# Vocabulary mismatch: grep vs semantic vs LLM search")
p()
p("**Question.** Vocabulary mismatch is the classic weakness of lexical search: the query and the")
p("answer use different words. How much does it actually cost grep/BM25, how well does semantic")
p("(embedding) search close the gap, and where do LLMs sit — depending on *how* the LLM is used")
p("(writing the query blind, vs. seeing the documents and query together)?")
p()
p("**Setup.** Corpus = 8,941 human-written passages from MS MARCO v2.1; queries = real typed Bing")
p("searches; each query's `is_selected` passage is its gold answer. Queries are bucketed by **stemmed")
p("term overlap** between the query and its gold passage — the vocabulary-mismatch axis:")
p()
p("- `zero` — no query word-stem appears in the answer (genuine mismatch). 104 queries (12%).")
p("- `low`  — <50% of query stems appear. 200 (22%).")
p("- `high` — >=50% appear. 600 (66%).")
p()
p("Stemming matters: without it, plurals (`psychopath`/`psychopaths`) masquerade as mismatch and")
p("make grep look artificially bad. All lexical arms are stemmed so the gap measured is *real*")
p("vocabulary mismatch (synonyms, paraphrase, acronyms), not morphology.")
p()

p("## Result 1 — full corpus, retrieval-only arms (n=904, 9k docs)")
p()
p("| arm | regime | hit@1 | hit@10 | MRR@10 | **zero** h@10 | low h@10 | high h@10 | lat/query | index |")
p("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
nm = {"grep": ("grep (naive ripgrep)", "lexical"), "bm25": ("BM25 (stemmed)", "lexical"),
      "potion": ("potion-32M (static emb)", "semantic"), "minilm": ("MiniLM (dense emb)", "semantic")}
for k in ["grep", "bm25", "potion", "minilm"]:
    o = dm[k]["overall"]; b = dm[k]["by_bucket"]; lat = dl[k]
    def h(x): return f"{b[x]['hit@10']:.2f}" if x in b else "-"
    name, reg = nm[k]
    p(f"| {name} | {reg} | {o['hit@1']:.3f} | {o['hit@10']:.3f} | {o['mrr@10']:.3f} | "
      f"**{h('zero')}** | {h('low')} | {h('high')} | {lat['per_query_ms']} ms | {lat['index_build_s']} s |")
p()
p(f"In the `zero` bucket, **grep and BM25 score exactly 0.00 hit@10** — a hard floor, by construction:")
p(f"they cannot match words that are not in the document. Dense MiniLM recovers")
p(f"**{dm['minilm']['by_bucket']['zero']['hit@10']:.0%}** of those same queries, and is *faster per query*")
p(f"(1.3 ms) than BM25 (2.2 ms) once indexed.")
p()

p(f"## Result 2 — all arms incl. LLM, same {N}-query subset (35 zero / 35 low / 20 high)")
p()
reg = {"grep (naive ripgrep)": "lexical", "bm25 (lexical, stemmed)": "lexical",
       "bm25 + LLM-expand [2a]": "LLM writes query (blind)", "claude-grep agentic [2b]": "LLM writes query (agentic)",
       "potion (semantic static)": "embeddings", "minilm (semantic dense)": "embeddings",
       "LLM-rerank / lexical cands [4a-lex]": "LLM reads docs+query", "LLM-rerank / semantic cands [4a-sem]": "LLM reads docs+query"}
p("| arm | regime | hit@1 | hit@10 | MRR@10 | **zero** h@10 | low h@10 | high h@10 |")
p("|---|---|--:|--:|--:|--:|--:|--:|")
for name, a in arms.items():
    o = a["overall"]; b = a["by_bucket"]
    def h(x): return f"{b[x]['hit@10']:.2f}" if x in b else "-"
    p(f"| {name} | {reg.get(name,'')} | {o['hit@1']:.3f} | {o['hit@10']:.3f} | {o['mrr@10']:.3f} | "
      f"**{h('zero')}** | {h('low')} | {h('high')} |")
p()

p("## Result 3 — first-stage recall@20 (what reranking can even see)")
p()
p("An LLM reranker can only re-order what the first-stage retriever handed it. % of golds present")
p("in the top-20 candidate pool:")
p()
p("| first stage | zero | low | high | all |")
p("|---|--:|--:|--:|--:|")
p("| BM25 (lexical)   | **0.00** | 0.69 | 1.00 | 0.49 |")
p("| MiniLM (semantic)| **0.86** | 0.94 | 1.00 | 0.92 |")
p()
p("This is why `rerank/lexical` scores 0.00 on the zero bucket — identical to BM25. The gold is")
p("never in the pool, so the LLM cannot recover it no matter how well it reads.")
p()

p("## Result 4 — latency & cost")
p()
p("| class | per-query latency | per-query cost | one-time |")
p("|---|---|---|---|")
p("| grep | ~6 ms | free | none |")
p("| BM25 | ~2 ms | free | 5.7 s index |")
p("| potion (static) | ~0.02 ms | free | 0.4 s index |")
p("| MiniLM (dense) | ~1.3 ms | free | 11 s index |")
p(f"| LLM arms | seconds–minutes | ~{use['avg_tokens_per_query_arm']:,} tokens (avg) | — |")
p()
p(f"The LLM arms together: {use['agents']} agents, {use['tokens_total']:,} tokens, {use['tool_uses']} tool")
p(f"calls, {use['wall_clock_s']:.0f} s wall-clock at ~16x concurrency. Agentic grep dominates (many")
p("iterative searches); rerank is mid; blind expansion is cheap. That is **~10,000x the per-query**")
p("**latency of dense retrieval** for, at best, a few points of hit@10.")
p()

p("## Findings")
p()
p("1. **Vocabulary mismatch is a hard zero for lexical search, not a soft penalty.** grep and BM25")
p("   get 0.00 hit@10 on genuine zero-overlap queries. The effect is real but *rare in a natural query")
p("   log* — only ~12% of MS MARCO queries are true zero-overlap — yet that slice is exactly where")
p("   lexical is blind.")
p()
p("2. **Your distinction holds and has a sharp boundary.** 'LLM sees documents + query together'")
p("   (rerank over semantic candidates: 0.86 on zero) beats 'LLM writes the query' — but the latter")
p("   splits by agency: blind one-shot expansion gets 0.54, while *agentic* grep (iterate, read hits,")
p("   guess synonyms) reaches 0.77. The more the LLM can see partial results and adapt, the more of")
p("   the gap it closes.")
p()
p("3. **The gap-closing power is in the retrieval, not the reading.** LLM rerank over *lexical*")
p("   candidates = 0.00 on zero (no better than grep); over *semantic* candidates = 0.86. Letting an")
p("   LLM read the documents does nothing for vocabulary mismatch if those documents were fetched")
p("   lexically. Rerank inherits the first stage's blindness.")
p()
p("4. **Plain dense embeddings are the value winner.** MiniLM matches agentic-grep on overall hit@10")
p("   (0.88), beats it on the zero bucket (0.83 vs 0.77), at ~1 ms and zero token cost per query vs")
p("   seconds and thousands of tokens. The only arm that clearly beats it (rerank-sem, 0.92) *uses")
p("   MiniLM as its first stage* and pays ~10,000x latency for +4 points.")
p()
p("**Bottom line.** For non-code text where queries are human-phrased, a small dense-embedding index")
p("closes most of the vocabulary-mismatch gap cheaply. An LLM helps further only when it reads")
p("semantically-retrieved candidates; an LLM that merely writes grep queries helps less, and an LLM")
p("reranking lexical results inherits grep's blind spot entirely.")
open("REPORT.md", "w").write("\n".join(L))
print("wrote REPORT.md (%d lines)" % len(L))
