"""Evaluate retrieval quality and tune chunk_size / top_k.

Usage:
    python -m eval.run_eval                      # keyword baseline (TF-IDF), no downloads
    python -m eval.run_eval --store chroma       # semantic embeddings
    python -m eval.run_eval --with-answers       # also score answers at the best config
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from statistics import mean

from app.agent import Agent, build_llm
from app.config import Settings
from app.ingest import ingest_directory
from app.store import build_store
from app.tools import load_lookup
from eval.metrics import f1, groundedness, keyword_hit, precision_at_k, recall_at_k

ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZES = [300, 600, 1000]
TOP_KS = [2, 4, 6]


def load_testset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retrieval(store_backend: str, docs_dir: Path, testset: list[dict],
                       chunk_size: int, top_k: int) -> dict:
    store = build_store(store_backend, None, f"eval_{chunk_size}")
    ingest_directory(store, docs_dir, chunk_size, min(100, chunk_size // 4))
    precisions, recalls = [], []
    for case in testset:
        sources = [r.source for r in store.search(case["question"], top_k)]
        relevant = set(case["relevant"])
        precisions.append(precision_at_k(sources, relevant, top_k))
        recalls.append(recall_at_k(sources, relevant, top_k))
    p, r = mean(precisions), mean(recalls)
    return {"chunk_size": chunk_size, "top_k": top_k, "precision": round(p, 3),
            "recall": round(r, 3), "f1": round(f1(p, r), 3)}


def evaluate_answers(store_backend: str, docs_dir: Path, testset: list[dict],
                     chunk_size: int, top_k: int) -> dict:
    settings = Settings.from_env()
    store = build_store(store_backend, None, "eval_answers")
    ingest_directory(store, docs_dir, chunk_size, min(100, chunk_size // 4))
    llm = build_llm(settings.anthropic_api_key, settings.model)
    agent = Agent(store, load_lookup(ROOT / "data/lookup.json"), llm, top_k)

    grounded, hits, mode = [], [], ""
    for case in testset:
        result = agent.ask(case["question"])
        mode = result.mode
        grounded.append(groundedness(result.answer, [r.text for r in result.retrieved]))
        hits.append(keyword_hit(result.answer, case["keywords"]))
    return {"mode": mode, "groundedness": round(mean(grounded), 3),
            "answer_accuracy": round(mean(hits), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", choices=["memory", "chroma"], default="memory")
    parser.add_argument("--with-answers", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "eval/results.json"))
    args = parser.parse_args()

    docs_dir = Path(os.getenv("DOCS_DIR", ROOT / "data/docs"))
    testset = load_testset(ROOT / "eval/testset.json")

    rows = [evaluate_retrieval(args.store, docs_dir, testset, cs, k)
            for cs, k in itertools.product(CHUNK_SIZES, TOP_KS)]

    print(f"Retrieval evaluation ({args.store}, {len(testset)} questions)\n")
    print(f"{'chunk':>6} {'top_k':>6} {'prec':>6} {'recall':>7} {'f1':>6}")
    for row in rows:
        print(f"{row['chunk_size']:>6} {row['top_k']:>6} {row['precision']:>6.3f} "
              f"{row['recall']:>7.3f} {row['f1']:>6.3f}")

    best = max(rows, key=lambda r: (r["f1"], r["recall"], -r["top_k"]))
    print(f"\nBest: chunk_size={best['chunk_size']}, top_k={best['top_k']} (F1 {best['f1']:.3f})")

    output = {"store": args.store, "sweep": rows, "best": best}
    if args.with_answers:
        answers = evaluate_answers(args.store, docs_dir, testset, best["chunk_size"], best["top_k"])
        output["answers"] = answers
        print(f"Answers ({answers['mode']}): groundedness {answers['groundedness']:.3f}, "
              f"accuracy {answers['answer_accuracy']:.3f}")

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
