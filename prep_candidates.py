"""Build candidate pools for the LLM rerank arms, from already-computed BM25 / MiniLM
rankings (results/det_results.json, top-25). For each query in the LLM subset we save the
top-20 candidates (doc_id + text) for two first stages:
    data/cand_bm25.json    lexical first stage
    data/cand_minilm.json  semantic first stage
Also reports first-stage recall@20 per bucket: the CEILING that rerank cannot exceed."""
import json
from collections import defaultdict

N = 20
corpus = {c["doc_id"]: c["text"] for c in (json.loads(l) for l in open("data/corpus.jsonl"))}
subset = json.load(open("data/llm_subset.json"))
det = json.load(open("results/det_results.json"))

def build(first_stage):
    out, recall = {}, defaultdict(lambda: [0, 0])
    for q in subset:
        qid = q["query_id"]
        cand_ids = det[first_stage][qid][:N]
        out[qid] = [{"doc_id": d, "text": corpus[d]} for d in cand_ids]
        gold = set(q["gold_doc_ids"])
        recall[q["bucket"]][1] += 1
        if gold & set(cand_ids):
            recall[q["bucket"]][0] += 1
    return out, recall

for fs, path in [("bm25", "data/cand_bm25.json"), ("minilm", "data/cand_minilm.json")]:
    out, recall = build(fs)
    json.dump(out, open(path, "w"))
    tot = [sum(recall[b][0] for b in recall), sum(recall[b][1] for b in recall)]
    print(f"\n{fs} first-stage recall@{N} (gold present in candidate pool):")
    for b in ["zero", "low", "high"]:
        if recall[b][1]:
            c, t = recall[b]
            print(f"   {b:5s}: {c:3d}/{t:<3d} = {c/t:.2f}")
    print(f"   ALL  : {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.2f}")
