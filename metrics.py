"""Ranking metrics. Each query has >=1 gold doc; rankings are lists of doc_ids."""
import math
from collections import defaultdict


def _rank_of_gold(ranking, gold):
    for i, d in enumerate(ranking):
        if d in gold:
            return i + 1  # 1-indexed
    return None


def per_query(ranking, gold):
    r = _rank_of_gold(ranking, gold)
    return {
        "hit@1": 1.0 if r == 1 else 0.0,
        "hit@3": 1.0 if (r and r <= 3) else 0.0,
        "hit@10": 1.0 if (r and r <= 10) else 0.0,
        "mrr@10": (1.0 / r) if (r and r <= 10) else 0.0,
        "ndcg@10": (1.0 / math.log2(r + 1)) if (r and r <= 10) else 0.0,
        "rank": r,
    }


def aggregate(rankings, queries):
    """rankings: {query_id: [doc_ids]}. queries: list of query dicts with gold+bucket.
    Returns overall and per-bucket means of each metric."""
    keys = ["hit@1", "hit@3", "hit@10", "mrr@10", "ndcg@10"]
    overall = defaultdict(list)
    by_bucket = defaultdict(lambda: defaultdict(list))
    for q in queries:
        qid = q["query_id"]
        if qid not in rankings:
            continue
        gold = set(q["gold_doc_ids"])
        m = per_query(rankings[qid], gold)
        for k in keys:
            overall[k].append(m[k])
            by_bucket[q["bucket"]][k].append(m[k])
    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0
    out = {"overall": {k: mean(overall[k]) for k in keys}, "n": len(overall["hit@1"])}
    out["by_bucket"] = {}
    for b in ["zero", "low", "high"]:
        if by_bucket[b]["hit@1"]:
            out["by_bucket"][b] = {k: mean(by_bucket[b][k]) for k in keys}
            out["by_bucket"][b]["n"] = len(by_bucket[b]["hit@1"])
    return out
