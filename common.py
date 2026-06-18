"""Shared tokenization / overlap utilities for the vocab-mismatch search benchmark."""
import re

# Small English stopword list (no nltk dependency).
STOP = set("""
a an the and or but if then else when at by for with about against between into through during
before after above below to from up down in out on off over under again further once here there
all any both each few more most other some such no nor not only own same so than too very s t can
will just don should now is are was were be been being have has had do does did of as it its this
that these those i you he she we they them his her their our your my me him us what which who whom
how why where whats what's how's i'm you're it's dont doesnt cant whats vs etc per also may might
""".split())

WORD_RE = re.compile(r"[a-z0-9]+")

import snowballstemmer
_STEM = snowballstemmer.stemmer("english")


def content_tokens(text):
    """Lowercase alphanumeric tokens with stopwords removed and len>1 (NO stemming).
    Used for the naive-grep arm and raw display."""
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOP and len(w) > 1]


def stem_tokens(text):
    """content_tokens but Porter-stemmed. Used for BM25 and for the overlap metric so
    that morphology (psychopath/psychopaths) is NOT counted as vocabulary mismatch."""
    return _STEM.stemWords(content_tokens(text))


def overlap_ratio(query, doc):
    """Fraction of the query's (stemmed) content terms that appear in the doc.
    0.0 => genuine vocabulary mismatch: no query word-stem is in the answer."""
    q = set(stem_tokens(query))
    if not q:
        return 0.0
    d = set(stem_tokens(doc))
    return len(q & d) / len(q)


def overlap_bucket(ratio):
    if ratio == 0.0:
        return "zero"      # pure vocabulary mismatch — lexical CANNOT match via query terms
    if ratio < 0.5:
        return "low"
    return "high"          # most query words appear verbatim in the answer
