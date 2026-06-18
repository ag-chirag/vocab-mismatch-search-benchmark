export const meta = {
  name: 'vocab-search-llm-arms',
  description: 'LLM search arms: blind query expansion, agentic grep, rerank over lexical/semantic candidates',
  phases: [
    { title: 'expand', detail: 'LLM writes synonyms/paraphrases blind (arm 2a)' },
    { title: 'rerank-lex', detail: 'LLM reranks top-20 BM25 candidates (arm 4a-lex)' },
    { title: 'rerank-sem', detail: 'LLM reranks top-20 MiniLM candidates (arm 4a-sem)' },
    { title: 'grep', detail: 'Claude greps the corpus iteratively (arm 2b)' },
  ],
}

const A = (typeof args === 'string' ? JSON.parse(args) : args) || {}
log('args parsed: ' + JSON.stringify(A))
const DIR = A.dir
const RANKED = { type: 'object', additionalProperties: false,
  properties: { query_id: { type: 'string' }, ranked: { type: 'array', items: { type: 'string' } } },
  required: ['query_id', 'ranked'] }
const RANKED_BATCH = { type: 'object', additionalProperties: false,
  properties: { results: { type: 'array', items: RANKED } }, required: ['results'] }
const EXPAND_BATCH = { type: 'object', additionalProperties: false,
  properties: { results: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: { query_id: { type: 'string' }, terms: { type: 'array', items: { type: 'string' } } },
    required: ['query_id', 'terms'] } } }, required: ['results'] }

// ---- arm 2a: blind query expansion (LLM never sees documents) ----
phase('expand')
const expandThunks = []
for (let i = 0; i < A.expandN; i++) {
  expandThunks.push(() => agent(
    `You help a keyword search engine beat vocabulary mismatch. Read the JSON file ` +
    `${DIR}/data/llm_batches/expand_${i}.json — it has items:[{query_id, query}]. ` +
    `You may NOT see any documents. For each query, using only world knowledge, output 6-15 extra ` +
    `search terms that are LIKELY to appear in a document that answers the query even when the query's ` +
    `own words do not: synonyms, paraphrases, alternative phrasings, acronym expansions/contractions, ` +
    `and closely related domain terms. Single words or short phrases. ` +
    `Write the JSON {"results":[{"query_id","terms":[...]}, ...]} to the file path given by the input's sibling ` +
    `"out" field (it is ${DIR}/results/llm_raw/expand_${i}.json) using the Write tool, then return the same object.`,
    { label: `expand_${i}`, phase: 'expand', schema: EXPAND_BATCH }))
}
const expandRes = await parallel(expandThunks)

// ---- arm 4a-lex / 4a-sem: rerank candidate passages (LLM sees docs + query) ----
function rerankThunks(tag, n) {
  const out = []
  for (let i = 0; i < n; i++) {
    out.push(() => agent(
      `You are a semantic reranker. Read ${DIR}/data/llm_batches/${tag}_${i}.json — it has ` +
      `items:[{query_id, query, candidates:[{doc_id, text}]}]. For each query, rank ALL its candidate ` +
      `doc_ids from most to least relevant — the passage that actually ANSWERS the query first. Judge by ` +
      `meaning, not word overlap (a passage can answer using completely different words). Return every ` +
      `candidate doc_id exactly once per query. Write {"results":[{"query_id","ranked":[doc_id,...]}, ...]} ` +
      `to ${DIR}/results/llm_raw/${tag}_${i}.json with the Write tool, then return the same object.`,
      { label: `${tag}_${i}`, phase: tag === 'rrlex' ? 'rerank-lex' : 'rerank-sem', schema: RANKED_BATCH }))
  }
  return out
}
phase('rerank-lex')
const rrlexRes = await parallel(rerankThunks('rrlex', A.rrlexN))
phase('rerank-sem')
const rrsemRes = await parallel(rerankThunks('rrsem', A.rrsemN))

// ---- arm 2b: agentic grep (Claude searches the corpus, iterating + guessing synonyms) ----
phase('grep')
const grepThunks = []
for (let i = 0; i < A.grepN; i++) {
  grepThunks.push(() => agent(
    `You must locate the passage that answers a query using ONLY keyword/regex search — like a developer ` +
    `using grep. Read ${DIR}/data/llm_batches/grep_${i}.json (fields: query, query_id, corpusTxt, rgPath, out). ` +
    `The corpus file corpusTxt has one passage per line formatted as  doc_id<TAB>passage_text . ` +
    `Find the doc_ids whose passage best answers the query. Rules: access the corpus ONLY through keyword/regex ` +
    `searches (use the Grep tool or run the ripgrep binary at rgPath via Bash on corpusTxt); you MAY iterate and ` +
    `try synonyms/alternate spellings/related terms; you may read individual lines your searches return; do NOT ` +
    `read or dump the whole corpus file. The doc_id is the text before the first TAB on a matching line. ` +
    `Return up to 10 doc_ids ranked best-first (most likely to answer the query first); [] if you find nothing. ` +
    `Write {"query_id","ranked":[doc_id,...]} to the "out" path with the Write tool, then return the same object.`,
    { label: `grep_${i}`, phase: 'grep', schema: RANKED }))
}
const grepRes = await parallel(grepThunks)

// ---- aggregate (backup channel; primary is the per-agent files on disk) ----
const expand = {}, rrlex = {}, rrsem = {}, grep = {}
for (const r of expandRes) if (r) for (const e of r.results) expand[e.query_id] = e.terms
for (const r of rrlexRes) if (r) for (const e of r.results) rrlex[e.query_id] = e.ranked
for (const r of rrsemRes) if (r) for (const e of r.results) rrsem[e.query_id] = e.ranked
for (const r of grepRes) if (r) grep[r.query_id] = r.ranked
log(`done: expand=${Object.keys(expand).length} rrlex=${Object.keys(rrlex).length} rrsem=${Object.keys(rrsem).length} grep=${Object.keys(grep).length}`)
return { counts: { expand: Object.keys(expand).length, rrlex: Object.keys(rrlex).length,
  rrsem: Object.keys(rrsem).length, grep: Object.keys(grep).length }, expand, rrlex, rrsem, grep }