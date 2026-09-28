"""Small text helpers shared by retrieval and evaluation."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it
    its may of on or our per should that the their them there these this to
    was we what when where which who why will with within you your""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lower-case word/number tokens, optionally without stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]
