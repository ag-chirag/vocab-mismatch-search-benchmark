"""Merge LLM-arm outputs with deterministic arms; compute every arm's metrics on the
SAME 90-query subset (fair comparison) + per-bucket breakdown. Writes results/final_metrics.json."""
import json, glob, os
import numpy as np
from common import stem_tokens
import metrics as M

corpus = [json.loads(l) for l in open("data/corpus.jsonl")]
queries = json.load(open("data/queries.json"))
subset = json.load(open("data/llm_subset.json"))
subset_ids = {q["query_id"] for q in subset}
by_id = {q["query_id"]: q for q in queries}
det = json.load(open("results/det_results.json"))

def load_raw(prefix):
    """Merge results/llm_raw/<prefix>_*.json -> {query_id: ranked or terms}."""
    out = {}
    for f in glob.glob(f"results/llm_raw/{prefix}_*.json"):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  WARN bad json {f}: {e}"); continue
        if "results" in d:               # batched (expand / rerank)
            for e in d["results"]:
                out[e["query_id"]] = e.get("ranked", e.get("terms"))
        elif "query_id" in d:            # single (grep)
            out[d["query_id"]] = d.get("ranked", [])
    return out

# ---- LLM arm outputs from disk ----
expand_terms = load_raw("expand")
claude_grep = load_raw("grep")
rerank_lex = load_raw("rrlex")
rerank_sem = load_raw("rrsem")

# ---- arm 2a: BM25 with LLM-expanded query (LLM writes the query, blind) ----
from rank_bm25 import BM25Okapi
toks = [stem_tokens(c["text"]) for c in corpus]
bm25 = BM25Okapi(toks)
ids = [c["doc_id"] for c in corpus]
bm25_expand = {}
for q in subset:
    qid = q["query_id"]
    aug = stem_tokens(q["query"])
    for t in expand_terms.get(qid, []):
        aug += stem_tokens(t)
    scores = bm25.get_scores(aug)
    top = np.argpartition(-scores, range(10))[:10]
    top = top[np.argsort(-scores[top])]
    bm25_expand[qid] = [ids[i] for i in top]

# ---- assemble rankings per arm, restricted to subset ----
def sub(d):
    return {qid: d.get(qid, []) for qid in subset_ids}

ARMS = {
    "grep (naive ripgrep)":        sub(det["grep"]),
    "bm25 (lexical, stemmed)":     sub(det["bm25"]),
    "bm25 + LLM-expand [2a]":      bm25_expand,
    "claude-grep agentic [2b]":    sub(claude_grep),
    "potion (semantic static)":    sub(det["potion"]),
    "minilm (semantic dense)":     sub(det["minilm"]),
    "LLM-rerank / lexical cands [4a-lex]": sub(rerank_lex),
    "LLM-rerank / semantic cands [4a-sem]": sub(rerank_sem),
}

final = {}
for name, rk in ARMS.items():
    final[name] = M.aggregate(rk, subset)
    cov = sum(1 for qid in subset_ids if rk.get(qid))
    final[name]["coverage"] = cov

# coverage report
print("LLM-arm coverage (queries with a non-empty answer / 90):")
print(f"  expand={len(expand_terms)} claude-grep={len(claude_grep)} "
      f"rerank-lex={len(rerank_lex)} rerank-sem={len(rerank_sem)}")

json.dump({"subset_n": len(subset), "arms": final}, open("results/final_metrics.json", "w"), indent=2)

ORDER = list(ARMS.keys())
print(f"\n=== SUBSET (n={len(subset)}: 35 zero / 35 low / 20 high) ===")
print(f"{'arm':38s} | {'hit@1':>5s} {'hit@10':>6s} {'mrr@10':>6s} | zero h@10 | low h@10 | high h@10")
for name in ORDER:
    a = final[name]; o = a["overall"]; b = a["by_bucket"]
    def h(x): return f"{b[x]['hit@10']:.2f}" if x in b else "  - "
    print(f"{name:38s} | {o['hit@1']:.3f} {o['hit@10']:.3f} {o['mrr@10']:.3f} | "
          f"  {h('zero')}   |  {h('low')}   |  {h('high')}")
