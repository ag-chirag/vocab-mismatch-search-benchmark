"""Deterministic (non-LLM) search arms + latency:
  1a  grep      real ripgrep, word-boundary, raw query terms (naive floor)
  1b  bm25      BM25Okapi over Porter-stemmed tokens (strong lexical baseline)
  3a  minilm    sentence-transformers all-MiniLM-L6-v2 (dense semantic)
  3b  potion    model2vec potion-retrieval-32M static embeddings (fast semantic)

Writes results/det_results.json, results/det_latency.json, results/det_metrics.json
"""
import json, time, subprocess, sys
import numpy as np
from common import content_tokens, stem_tokens
import metrics as M

TOPK = 25  # store deeper rankings so LLM-rerank arms can reuse top-20 candidate pools
# Real ripgrep binary (the shell `rg` here is a Claude wrapper function, not exec-able).
import os
RG = next((p for p in [
    "/Applications/Visual Studio Code.app/Contents/Resources/app/node_modules/@vscode/ripgrep/bin/rg",
    "/opt/homebrew/bin/rg", "/usr/local/bin/rg",
] if os.path.exists(p)), "rg")

def load():
    corpus = [json.loads(l) for l in open("data/corpus.jsonl")]
    queries = json.load(open("data/queries.json"))
    return corpus, queries

# ---------- 1a: real ripgrep ----------
def arm_grep(corpus, queries):
    # lineno (1-based) in corpus.txt == index in corpus list + 1
    lineno_to_doc = {i + 1: c["doc_id"] for i, c in enumerate(corpus)}
    rankings, times = {}, []
    for q in queries:
        terms = list(dict.fromkeys(content_tokens(q["query"])))  # unique, order-preserving
        if not terms:
            rankings[q["query_id"]] = []
            continue
        args = [RG, "-i", "-w", "-o", "-n"]
        for t in terms:
            args += ["-e", t]
        args += ["--", "data/corpus.txt"]  # note: -F not needed (terms are [a-z0-9]+)
        # ripgrep treats -e as regex; our terms are alphanumeric so safe, but be explicit:
        t0 = time.perf_counter()
        proc = subprocess.run(args, capture_output=True, text=True)
        times.append(time.perf_counter() - t0)
        distinct, total = {}, {}
        for line in proc.stdout.splitlines():
            # format: LINENO:match
            ci = line.find(":")
            if ci < 0:
                continue
            try:
                ln = int(line[:ci])
            except ValueError:
                continue
            match = line[ci + 1:].lower()
            distinct.setdefault(ln, set()).add(match)
            total[ln] = total.get(ln, 0) + 1
        scored = sorted(distinct.keys(), key=lambda ln: (len(distinct[ln]), total[ln]), reverse=True)
        rankings[q["query_id"]] = [lineno_to_doc[ln] for ln in scored[:TOPK]]
    return rankings, {"per_query_ms": round(1000 * np.mean(times), 2),
                      "p50_ms": round(1000 * np.median(times), 2),
                      "index_build_s": 0.0, "note": "no index; ripgrep scans the corpus file each query"}

# ---------- 1b: BM25 ----------
def arm_bm25(corpus, queries):
    from rank_bm25 import BM25Okapi
    t0 = time.perf_counter()
    toks = [stem_tokens(c["text"]) for c in corpus]
    bm25 = BM25Okapi(toks)
    build = time.perf_counter() - t0
    ids = [c["doc_id"] for c in corpus]
    rankings, times = {}, []
    for q in queries:
        qt = stem_tokens(q["query"])
        t1 = time.perf_counter()
        scores = bm25.get_scores(qt)
        top = np.argpartition(-scores, range(min(TOPK, len(scores))))[:TOPK]
        top = top[np.argsort(-scores[top])]
        times.append(time.perf_counter() - t1)
        rankings[q["query_id"]] = [ids[i] for i in top]
    return rankings, {"per_query_ms": round(1000 * np.mean(times), 2),
                      "p50_ms": round(1000 * np.median(times), 2),
                      "index_build_s": round(build, 2)}

# ---------- dense helpers ----------
def _dense(corpus, queries, encode_corpus, encode_queries, label):
    ids = [c["doc_id"] for c in corpus]
    t0 = time.perf_counter()
    C = encode_corpus([c["text"] for c in corpus])
    build = time.perf_counter() - t0
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    qtexts = [q["query"] for q in queries]
    t1 = time.perf_counter()
    Q = encode_queries(qtexts)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    sims = np.nan_to_num(Q @ C.T)  # (nq, nd); guard stray NaN/inf from BLAS
    enc_search = time.perf_counter() - t1
    rankings = {}
    for i, q in enumerate(queries):
        top = np.argpartition(-sims[i], range(TOPK))[:TOPK]
        top = top[np.argsort(-sims[i][top])]
        rankings[q["query_id"]] = [ids[j] for j in top]
    return rankings, {"per_query_ms": round(1000 * enc_search / len(queries), 2),
                      "index_build_s": round(build, 2),
                      "note": f"{label}: per_query_ms = (encode query + full cosine scan)/nq"}

def arm_minilm(corpus, queries):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda xs: np.asarray(m.encode(xs, batch_size=128, show_progress_bar=False, convert_to_numpy=True))
    return _dense(corpus, queries, enc, enc, "minilm")

def arm_potion(corpus, queries):
    from model2vec import StaticModel
    m = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    enc = lambda xs: np.asarray(m.encode(xs))
    return _dense(corpus, queries, enc, enc, "potion")

ARMS = {"grep": arm_grep, "bm25": arm_bm25, "minilm": arm_minilm, "potion": arm_potion}

if __name__ == "__main__":
    which = sys.argv[1:] or list(ARMS)
    corpus, queries = load()
    print(f"corpus={len(corpus)} queries={len(queries)}  arms={which}", flush=True)
    results = json.load(open("results/det_results.json")) if __import__("os").path.exists("results/det_results.json") else {}
    latency = json.load(open("results/det_latency.json")) if __import__("os").path.exists("results/det_latency.json") else {}
    mets = json.load(open("results/det_metrics.json")) if __import__("os").path.exists("results/det_metrics.json") else {}
    for name in which:
        t = time.perf_counter()
        r, lat = ARMS[name](corpus, queries)
        results[name] = r
        latency[name] = lat
        mets[name] = M.aggregate(r, queries)
        json.dump(results, open("results/det_results.json", "w"))
        json.dump(latency, open("results/det_latency.json", "w"), indent=2)
        json.dump(mets, open("results/det_metrics.json", "w"), indent=2)
        o = mets[name]["overall"]
        print(f"[{name:7s}] {time.perf_counter()-t:5.1f}s  hit@1={o['hit@1']:.3f} hit@10={o['hit@10']:.3f} "
              f"mrr@10={o['mrr@10']:.3f}  lat/q={lat['per_query_ms']}ms  build={lat['index_build_s']}s", flush=True)
    print("DONE")
