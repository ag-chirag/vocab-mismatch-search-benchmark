"""Render figures/bucket_hit10.png and figures/pareto.png from results/final_metrics.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)
fm = json.load(open("results/final_metrics.json"))["arms"]

ORDER = [
    ("grep (naive ripgrep)", "grep"),
    ("bm25 (lexical, stemmed)", "BM25"),
    ("bm25 + LLM-expand [2a]", "BM25+expand"),
    ("claude-grep agentic [2b]", "claude-grep"),
    ("potion (semantic static)", "potion"),
    ("minilm (semantic dense)", "MiniLM"),
    ("LLM-rerank / lexical cands [4a-lex]", "rerank-lex"),
    ("LLM-rerank / semantic cands [4a-sem]", "rerank-sem"),
]
short = [s for _, s in ORDER]

def h10(name, b):
    bb = fm[name]["by_bucket"]
    return bb[b]["hit@10"] if b in bb else 0.0

# ---- Figure 1: grouped bar, hit@10 by overlap bucket ----
zero = [h10(n, "zero") for n, _ in ORDER]
low = [h10(n, "low") for n, _ in ORDER]
high = [h10(n, "high") for n, _ in ORDER]
x = np.arange(len(short)); w = 0.27
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(x - w, zero, w, label="zero overlap (true mismatch)", color="#D85A30")
ax.bar(x, low, w, label="low", color="#EF9F27")
ax.bar(x + w, high, w, label="high", color="#1D9E75")
ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("hit@10"); ax.set_ylim(0, 1.05)
ax.set_title("Retrieval accuracy by query–answer vocabulary overlap", fontsize=11)
ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.25)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout(); fig.savefig("figures/bucket_hit10.png", dpi=150); plt.close(fig)

# ---- Figure 2: accuracy vs latency Pareto (LLM latencies approximate) ----
pts = [
    ("grep", 6, "overall", "#888780"), ("BM25", 2, "overall", "#888780"),
    ("potion", 0.02, "overall", "#1D9E75"), ("MiniLM", 1.3, "overall", "#1D9E75"),
    ("BM25+expand", 4000, "overall", "#BA7517"), ("claude-grep", 60000, "overall", "#BA7517"),
    ("rerank-lex", 12000, "overall", "#7F77DD"), ("rerank-sem", 12000, "overall", "#7F77DD"),
]
name_map = {s: n for n, s in ORDER}
fig, ax = plt.subplots(figsize=(8, 4.4))
for s, lat, _, c in pts:
    acc = fm[name_map[s]]["overall"]["hit@10"]
    ax.scatter(lat, acc, s=70, color=c, zorder=3)
    dx = -10 if s in ("rerank-lex", "claude-grep", "BM25+expand") else 1.25
    ha = "right" if dx < 0 else "left"
    ax.annotate(s, (lat, acc), xytext=(lat * dx if dx > 0 else lat / 10, acc + 0.015),
                fontsize=8, ha=ha)
ax.set_xscale("log"); ax.set_xlim(0.008, 200000); ax.set_ylim(0.25, 1.0)
ax.set_xlabel("per-query latency (ms, log) — LLM values approximate")
ax.set_ylabel("overall hit@10")
ax.set_title("Accuracy vs. latency", fontsize=11)
ax.grid(alpha=0.25)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=l)
       for c, l in [("#888780", "lexical"), ("#1D9E75", "embeddings"),
                    ("#BA7517", "LLM writes query"), ("#7F77DD", "LLM reads docs+query")]]
ax.legend(handles=leg, fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("figures/pareto.png", dpi=150); plt.close(fig)
print("wrote figures/bucket_hit10.png and figures/pareto.png")
