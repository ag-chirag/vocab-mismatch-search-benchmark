"""Build a shared retrieval corpus from MS MARCO v2.1 (real typed Bing queries +
human-written web passages). We pool every passage across the sampled queries into
one corpus; each query's is_selected passage(s) are its gold relevance labels.

Output:
  data/corpus.jsonl   one {doc_id, text, url} per line
  data/corpus.txt     doc_id<TAB>single-line-text  (for ripgrep arm)
  data/queries.json   [{query_id, query, gold_doc_ids, gold_overlap, bucket}]
"""
import json, random, sys, hashlib
from datasets import load_dataset
from common import overlap_ratio, overlap_bucket, content_tokens

random.seed(13)
# Per-bucket caps to oversample the rare-but-critical vocabulary-mismatch cases.
CAPS = {"zero": 130, "low": 200, "high": 600}
MAX_SCAN = 30000

def norm_line(t):
    return " ".join(t.split())

def pid(text):
    return "d" + hashlib.md5(text.encode()).hexdigest()[:12]

print(f"Streaming MS MARCO v2.1 validation; per-bucket caps {CAPS}, max scan {MAX_SCAN}...", flush=True)
ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)

corpus = {}            # doc_id -> {text, url}
queries = []
from collections import Counter
kept = Counter()
seen_q = 0
for ex in ds:
    seen_q += 1
    if seen_q > MAX_SCAN or all(kept[b] >= CAPS[b] for b in CAPS):
        break
    p = ex["passages"]
    texts, sel, urls = p["passage_text"], p["is_selected"], p["url"]
    gold_idx = [i for i, s in enumerate(sel) if s == 1]
    # Require exactly one selected passage => unambiguous gold.
    if len(gold_idx) != 1:
        continue
    q = ex["query"].strip()
    if len(q) < 8:
        continue
    if len(content_tokens(q)) < 1:
        continue  # degenerate query (all stopwords / single-char tokens, e.g. "c# what is it")
    # Compute bucket on the gold passage first; skip if that bucket is full.
    gtext = texts[gold_idx[0]].strip()
    if not gtext:
        continue
    b = overlap_bucket(overlap_ratio(q, gtext))
    if kept[b] >= CAPS[b]:
        continue
    gold_doc_ids = []
    for i, txt in enumerate(texts):
        txt = txt.strip()
        if not txt:
            continue
        did = pid(txt)
        if did not in corpus:
            corpus[did] = {"text": txt, "url": urls[i]}
        if i in gold_idx:
            gold_doc_ids.append(did)
    if not gold_doc_ids:
        continue
    r = overlap_ratio(q, corpus[gold_doc_ids[0]]["text"])
    queries.append({
        "query_id": str(ex["query_id"]),
        "query": q,
        "gold_doc_ids": gold_doc_ids,
        "gold_overlap": round(r, 3),
        "bucket": b,
        "query_type": ex.get("query_type"),
    })
    kept[b] += 1

# Write corpus
with open("data/corpus.jsonl", "w") as f:
    for did, v in corpus.items():
        f.write(json.dumps({"doc_id": did, "text": v["text"], "url": v["url"]}) + "\n")
with open("data/corpus.txt", "w") as f:
    for did, v in corpus.items():
        f.write(f"{did}\t{norm_line(v['text'])}\n")
with open("data/queries.json", "w") as f:
    json.dump(queries, f, indent=2)

# Stratified subset for the (expensive) LLM arms: oversample the mismatch region.
by_b = {"zero": [], "low": [], "high": []}
for q in queries:
    by_b[q["bucket"]].append(q)
for b in by_b:
    random.shuffle(by_b[b])
llm_subset = by_b["zero"][:35] + by_b["low"][:35] + by_b["high"][:20]
with open("data/llm_subset.json", "w") as f:
    json.dump(llm_subset, f, indent=2)

bc = Counter(q["bucket"] for q in queries)
print(f"\nScanned {seen_q} MS MARCO examples.")
print(f"Corpus: {len(corpus)} unique passages")
print(f"Queries: {len(queries)}")
print(f"Overlap buckets (query-term coverage of the gold passage):")
for b in ["zero", "low", "high"]:
    print(f"   {b:5s}: {bc[b]:4d}  ({100*bc[b]/len(queries):.1f}%)")
print(f"Median gold passage chars: {sorted(len(v['text']) for v in corpus.values())[len(corpus)//2]}")
sc = Counter(q["bucket"] for q in llm_subset)
print(f"LLM subset: {len(llm_subset)}  (zero={sc['zero']} low={sc['low']} high={sc['high']})")
print("DONE")
