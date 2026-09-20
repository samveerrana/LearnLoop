from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit(suite_path: Path, summary_path: Path) -> dict:
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results_path = Path(summary["results"])
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {case["id"]: case for case in suite["cases"]}
    ids = [row.get("id") for row in rows]
    errors: list[str] = []
    suite_hash = hashlib.sha256(suite_bytes).hexdigest()
    if summary.get("suite_sha256") != suite_hash:
        errors.append("summary suite hash does not match suite bytes")
    if len(ids) != len(set(ids)):
        errors.append("duplicate result IDs")
    if set(ids) != set(expected):
        errors.append("result IDs do not exactly cover frozen suite")
    for row in rows:
        case = expected.get(row.get("id"))
        if case is None:
            continue
        if row.get("expected") != case["expected"]:
            errors.append(f"{row['id']}: expected answer differs from suite")
        if row.get("category") != case["category"]:
            errors.append(f"{row['id']}: category differs from suite")
        recomputed = row.get("answer") == case["expected"]
        if row.get("correct") is not recomputed:
            errors.append(f"{row['id']}: incorrect correctness flag")
    correct = sum(bool(row.get("correct")) for row in rows)
    if summary.get("correct") != correct or summary.get("total") != len(rows):
        errors.append("summary totals do not match raw rows")
    for category, values in summary.get("categories", {}).items():
        subset = [row for row in rows if row.get("category") == category]
        if values != {"correct": sum(bool(row.get("correct")) for row in subset), "total": len(subset)}:
            errors.append(f"{category}: category totals do not match raw rows")
    observed_features = {
        "verified_tools": any(
            row.get("used") in {"verified-local-tool", "model-planned-verified-calculator"}
            for row in rows
        ),
        "temporary_parameter_rewrite": any(
            row.get("used") == "temporary-parameter-rewrite" for row in rows
        ),
        "official_retrieval": any(
            isinstance(row.get("training_trace"), dict)
            and row["training_trace"].get("retrieval_api") in {
                "Wikipedia MediaWiki API", "Hugging Face Dataset Viewer API"
            }
            and row["training_trace"].get("scraping") is False
            for row in rows
        ),
    }
    return {
        "passed": not errors, "suite_sha256": suite_hash, "summary": str(summary_path),
        "results": str(results_path), "rows": len(rows), "correct": correct,
        "observed_features": observed_features, "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit a completed LearnLoop benchmark run")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.suite, args.summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
