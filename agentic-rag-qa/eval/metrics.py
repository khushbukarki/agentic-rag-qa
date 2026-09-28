"""Retrieval and answer-quality metrics."""
from __future__ import annotations

import re

from app.text import split_sentences, tokenize

_CITATION_RE = re.compile(r"\[[^\]]*\]")


def precision_at_k(retrieved_sources: list[str], relevant: set[str], k: int) -> float:
    """Share of the top-k retrieved chunks that come from a relevant document."""
    top = retrieved_sources[:k]
    if not top:
        return 0.0
    return sum(s in relevant for s in top) / len(top)


def recall_at_k(retrieved_sources: list[str], relevant: set[str], k: int) -> float:
    """Share of relevant documents that appear at least once in the top-k."""
    if not relevant:
        return 1.0
    return len(set(retrieved_sources[:k]) & relevant) / len(relevant)


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def groundedness(answer: str, contexts: list[str], threshold: float = 0.6) -> float:
    """Share of answer sentences whose content words mostly appear in the retrieved context.

    A lightweight lexical proxy for "is the answer supported by the sources?".
    It won't catch subtle paraphrase, but it reliably flags answers that
    introduce facts (numbers, names) that were never retrieved.
    """
    context_tokens = set(tokenize(" ".join(contexts)))
    scored = 0
    supported = 0
    for sentence in split_sentences(answer):
        tokens = tokenize(_CITATION_RE.sub(" ", sentence))
        if not tokens:
            continue
        scored += 1
        if sum(t in context_tokens for t in tokens) / len(tokens) >= threshold:
            supported += 1
    return supported / scored if scored else 0.0


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    text = answer.lower()
    return all(k.lower() in text for k in keywords)
