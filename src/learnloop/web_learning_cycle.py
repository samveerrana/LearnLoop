from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .memory import MemoryStore
from .research import WikipediaResearcher
from .web_dataset import build_licensed_web_dataset
from .web_weight_eval import evaluate


def compare(base: dict, rewritten: dict) -> dict:
    gain = rewritten["correct"] - base["correct"]
    return {
        "base_correct": base["correct"], "rewritten_correct": rewritten["correct"],
        "total": base["total"], "absolute_gain_questions": gain,
        "percentage_point_gain": round(100 * gain / base["total"], 2),
        "task_score_multiplier": round(rewritten["correct"] / base["correct"], 3) if base["correct"] else None,
        "accepted": gain > 0,
        "claim_limit": "Narrow closed-book fact retention only; not general model-size equivalence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a gated official-web temporary-weight learning cycle")
    parser.add_argument("topic")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("training/web-cycle"))
    parser.add_argument("--report-dir", type=Path, default=Path("benchmarks/web-cycle"))
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--keep-capsule", action="store_true", help="Keep an accepted capsule for inspection")
    args = parser.parse_args()
    data, capsule = args.work / "licensed-data", args.work / "temporary-capsule"
    if args.work.exists():
        raise SystemExit(f"refusing to overwrite existing cycle directory: {args.work}")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="learnloop-official-api-") as temporary:
        memory = MemoryStore(Path(temporary) / "research.sqlite3")
        retrieval = WikipediaResearcher(memory).research(args.topic)
        counts = build_licensed_web_dataset(memory.search_knowledge(args.topic, limit=5), data)

    base = evaluate(args.model, data / "closed-book-test.jsonl", args.report_dir / "base.json")
    subprocess.run([
        sys.executable, "-m", "learnloop.train_capsule", "--model", str(args.model),
        "--data", str(data), "--output", str(capsule), "--iters", str(args.iters),
        "--layers", "36", "--rank", "8", "--learning-rate", "0.00001",
    ], check=True)
    rewritten = evaluate(args.model, data / "closed-book-test.jsonl", args.report_dir / "rewritten.json", capsule)
    result = {
        "topic": args.topic, "retrieval": "Wikipedia MediaWiki API", "scraping": False,
        "pages": retrieval.pages_saved, "dataset": counts, **compare(base, rewritten),
    }
    args.report_dir.joinpath("comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    keep = result["accepted"] and args.keep_capsule
    if not keep:
        shutil.rmtree(capsule)
    shutil.rmtree(data)
    if args.work.exists() and not any(args.work.iterdir()):
        args.work.rmdir()
    result["temporary_capsule_deleted"] = not keep
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
