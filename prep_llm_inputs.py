"""Prepare per-agent input files for the LLM-arm workflow, plus a tiny args dict.

LLM arms (on the 90-query llm_subset):
  2a expand     agent sees ONLY the query -> synonyms/paraphrase/acronym expansions
                (Python then re-runs BM25 with the augmented query)   "LLM writes the query"
  2b grep       agent greps the corpus iteratively -> ranked doc_ids   "Claude's grep"
  4a rerank-lex agent reads query + top-20 BM25 candidates -> reranks   "LLM sees docs+query"
  4a rerank-sem agent reads query + top-20 MiniLM candidates -> reranks  (semantic 1st stage)

Each agent reads ONE small file data/llm_batches/<arm>_<i>.json and writes
results/llm_raw/<arm>_<i>.json. The workflow only needs counts + the project dir.
"""
import json, os

ABS = os.path.abspath(".")
subset = json.load(open("data/llm_subset.json"))
cand_bm25 = json.load(open("data/cand_bm25.json"))
cand_minilm = json.load(open("data/cand_minilm.json"))
os.makedirs("data/llm_batches", exist_ok=True)
os.makedirs("results/llm_raw", exist_ok=True)
RG = "/Applications/Visual Studio Code.app/Contents/Resources/app/node_modules/@vscode/ripgrep/bin/rg"

def chunks(xs, n):
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def w(path, obj):
    json.dump(obj, open(path, "w"))

# 2b grep: one file per query
for i, q in enumerate(subset):
    w(f"data/llm_batches/grep_{i}.json", {
        "query_id": q["query_id"], "query": q["query"],
        "corpusTxt": f"{ABS}/data/corpus.txt", "rgPath": RG,
        "out": f"{ABS}/results/llm_raw/grep_{i}.json"})
grepN = len(subset)

# 2a expand: 3 batches of 30
exp = chunks(subset, 30)
for i, ch in enumerate(exp):
    w(f"data/llm_batches/expand_{i}.json", {
        "items": [{"query_id": q["query_id"], "query": q["query"]} for q in ch],
        "out": f"{ABS}/results/llm_raw/expand_{i}.json"})

# 4a rerank: batches of 6, with candidates inline
def rr(cands, tag):
    chs = chunks(subset, 6)
    for i, ch in enumerate(chs):
        w(f"data/llm_batches/{tag}_{i}.json", {
            "items": [{"query_id": q["query_id"], "query": q["query"],
                       "candidates": cands[q["query_id"]]} for q in ch],
            "out": f"{ABS}/results/llm_raw/{tag}_{i}.json"})
    return len(chs)

rrlexN = rr(cand_bm25, "rrlex")
rrsemN = rr(cand_minilm, "rrsem")

args = {"dir": ABS, "grepN": grepN, "expandN": len(exp), "rrlexN": rrlexN, "rrsemN": rrsemN}
w("data/args.json", args)
print(json.dumps(args, indent=2))
