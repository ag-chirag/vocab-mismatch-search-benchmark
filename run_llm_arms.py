"""Standalone reproduction of the four LLM search arms via the Anthropic API.

This is a portable port of workflow/llm_arms_workflow.js (the Claude Code multi-agent
workflow that produced the published numbers). Prompts are kept identical in spirit.
It reads the per-batch inputs written by prep_llm_inputs.py and writes the same
results/llm_raw/*.json files that analyze.py consumes.

Requires:  pip install anthropic   and   ANTHROPIC_API_KEY in the environment.
Model:     defaults to claude-opus-4-8 (what the original run used); override with MODEL.
Note:      the original results were produced by the Claude Code workflow; this script
           reproduces the same protocol but was not re-run to regenerate the committed
           numbers. Expect small run-to-run variation (LLMs are nondeterministic).
"""
import json, glob, os, subprocess
from anthropic import Anthropic

client = Anthropic()
MODEL = os.environ.get("MODEL", "claude-opus-4-8")
RAW = "results/llm_raw"
os.makedirs(RAW, exist_ok=True)


def submit_tool(props, required):
    return [{"name": "submit", "description": "Submit the final result.",
             "input_schema": {"type": "object", "properties": props, "required": required}}]


def one_shot(prompt, tool):
    """Force a single structured tool call and return its input dict."""
    r = client.messages.create(model=MODEL, max_tokens=4096, tools=tool,
                               tool_choice={"type": "tool", "name": "submit"},
                               messages=[{"role": "user", "content": prompt}])
    for b in r.content:
        if b.type == "tool_use":
            return b.input
    return {}


# ---------- arm 2a: blind query expansion ----------
def run_expand():
    tool = submit_tool({"results": {"type": "array", "items": {"type": "object",
            "properties": {"query_id": {"type": "string"},
                           "terms": {"type": "array", "items": {"type": "string"}}},
            "required": ["query_id", "terms"]}}}, ["results"])
    for f in sorted(glob.glob("data/llm_batches/expand_*.json")):
        d = json.load(open(f))
        prompt = ("You help a keyword search engine beat vocabulary mismatch. Here are queries:\n"
                  + json.dumps(d["items"]) +
                  "\nYou may NOT see any documents. For each query, using only world knowledge, output "
                  "6-15 extra search terms LIKELY to appear in a document that answers the query even when "
                  "the query's own words do not: synonyms, paraphrases, alternative phrasings, acronym "
                  "expansions/contractions, related domain terms. Call submit with results:[{query_id, terms}].")
        json.dump(one_shot(prompt, tool), open(d["out"], "w"))
        print("expand", os.path.basename(d["out"]))


# ---------- arm 4a: rerank candidates ----------
def run_rerank(tag):
    tool = submit_tool({"results": {"type": "array", "items": {"type": "object",
            "properties": {"query_id": {"type": "string"},
                           "ranked": {"type": "array", "items": {"type": "string"}}},
            "required": ["query_id", "ranked"]}}}, ["results"])
    for f in sorted(glob.glob(f"data/llm_batches/{tag}_*.json")):
        d = json.load(open(f))
        prompt = ("You are a semantic reranker. For each query, rank ALL its candidate doc_ids from most "
                  "to least relevant — the passage that actually ANSWERS the query first. Judge by meaning, "
                  "not word overlap. Return every candidate doc_id exactly once per query.\n"
                  + json.dumps(d["items"]) +
                  "\nCall submit with results:[{query_id, ranked:[doc_id,...]}].")
        json.dump(one_shot(prompt, tool), open(d["out"], "w"))
        print(tag, os.path.basename(d["out"]))


# ---------- arm 2b: agentic grep ----------
def ripgrep(pattern, corpus, rg, ignorecase=True, word=False, max_lines=40):
    args = [rg if os.path.exists(rg) else "rg", "-n"]
    if ignorecase:
        args.append("-i")
    if word:
        args.append("-w")
    args += ["-e", pattern, "--", corpus]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=20).stdout
    except Exception as e:
        return f"(search error: {e})"
    lines = out.splitlines()
    head = "\n".join(lines[:max_lines])
    return head + (f"\n... (+{len(lines)-max_lines} more lines)" if len(lines) > max_lines else "") or "(no matches)"


def run_grep(max_rounds=8):
    tools = [
        {"name": "ripgrep", "description": "Search the corpus file with a regex; returns matching lines.",
         "input_schema": {"type": "object", "properties": {
             "pattern": {"type": "string"}, "word": {"type": "boolean"}}, "required": ["pattern"]}},
        {"name": "submit", "description": "Submit ranked doc_ids best-first.",
         "input_schema": {"type": "object", "properties": {
             "ranked": {"type": "array", "items": {"type": "string"}}}, "required": ["ranked"]}},
    ]
    for f in sorted(glob.glob("data/llm_batches/grep_*.json")):
        d = json.load(open(f))
        sys = ("You must locate the passage that answers a query using ONLY keyword/regex search, like a "
               "developer using grep. The corpus file has one passage per line as  doc_id<TAB>passage_text . "
               "The doc_id is the text before the first TAB on a matching line. You MAY iterate and try "
               "synonyms/alternate spellings/related terms; do NOT dump the whole file. When confident, call "
               "submit with up to 10 doc_ids ranked best-first (most likely to answer first); [] if nothing.")
        msgs = [{"role": "user", "content":
                 f"query: {d['query']!r}\ncorpus file: {d['corpusTxt']}\nUse the ripgrep tool to search it."}]
        ranked = []
        for _ in range(max_rounds):
            r = client.messages.create(model=MODEL, max_tokens=1500, system=sys, tools=tools, messages=msgs)
            msgs.append({"role": "assistant", "content": r.content})
            results, done = [], False
            for b in r.content:
                if b.type == "tool_use" and b.name == "ripgrep":
                    obs = ripgrep(b.input["pattern"], d["corpusTxt"], d["rgPath"], word=b.input.get("word", False))
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": obs})
                elif b.type == "tool_use" and b.name == "submit":
                    ranked = b.input.get("ranked", [])
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": "ok"})
                    done = True
            if results:
                msgs.append({"role": "user", "content": results})
            if done or r.stop_reason != "tool_use":
                break
        json.dump({"query_id": d["query_id"], "ranked": ranked[:10]}, open(d["out"], "w"))
        print("grep", os.path.basename(d["out"]), "->", len(ranked), "ids")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["expand", "rerank", "grep"]
    if "expand" in which:
        run_expand()
    if "rerank" in which:
        run_rerank("rrlex"); run_rerank("rrsem")
    if "grep" in which:
        run_grep()
    print("done — now run: python analyze.py")
