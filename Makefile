PY ?= python

.PHONY: all corpus deterministic candidates llm-inputs llm-arms analyze figures report clean

# Full pipeline. The llm-arms step needs ANTHROPIC_API_KEY (or run the Claude Code workflow instead).
all: corpus deterministic candidates llm-inputs llm-arms analyze figures report

corpus:        ; $(PY) build_corpus.py
deterministic: ; $(PY) arms_deterministic.py
candidates:    ; $(PY) prep_candidates.py
llm-inputs:    ; $(PY) prep_llm_inputs.py
llm-arms:      ; $(PY) run_llm_arms.py        # requires ANTHROPIC_API_KEY
analyze:       ; $(PY) analyze.py
figures:       ; $(PY) make_figures.py
report:        ; $(PY) report.py

# Everything except the LLM arms (no API key needed) — reproduces the deterministic results.
deterministic-only: corpus deterministic candidates analyze figures report

clean:
	rm -rf data results/llm_raw results/det_results.json figures __pycache__
