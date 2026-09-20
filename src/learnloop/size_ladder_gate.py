from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def evaluate(
    comparisons: list[dict],
    base_parameters_b: float = 4.022,
    reference_parameters_b: float = 7.6,
    reference_name: str = "Qwen2.5 7.6B",
) -> dict:
    distinct = len({item["suite_sha256"] for item in comparisons}) == len(comparisons)
    checks = {
        "reference_is_approximately_twice_base_size": reference_parameters_b / base_parameters_b >= 1.8,
        "at_least_three_distinct_evaluations": len(comparisons) >= 3 and distinct,
        "candidate_matches_or_beats_reference_every_time": all(item["candidate"] >= item["reference"] for item in comparisons),
        "all_artifacts_verified": all(item.get("verified") is True for item in comparisons),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "base_parameters_b": base_parameters_b, "reference_parameters_b": reference_parameters_b,
        "parameter_ratio": round(reference_parameters_b / base_parameters_b, 3),
        "comparisons": comparisons,
        "claim": (
            f"The LearnLoop 4B framework consistently outperformed the tested {reference_name} "
            "checkpoint on these evaluations; this is not universal 8B equivalence."
        ),
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fresh_score(path: Path) -> tuple[int, int]:
    report = _load(path)
    correct = sum(row.get("answer") == row.get("expected") for row in report["rows"])
    if correct != report["correct"] or len(report["rows"]) != report["total"]:
        raise ValueError(f"raw report totals do not verify: {path}")
    return correct, len(report["rows"])


def build(
    root: Path,
    reference_slug: str = "qwen2.5-7b",
    reference_name: str = "Qwen2.5 7.6B",
    reference_parameters_b: float = 7.6,
) -> dict:
    v3_candidate = _load(root / "benchmarks/benchmark-100-v3/audit-candidate.json")
    v3_reference = _load(root / f"benchmarks/benchmark-100-v3/audit-reference-{reference_slug}.json")
    v5_candidate = _load(root / "benchmarks/benchmark-100-v5/audit-candidate.json")
    v5_reference = _load(root / f"benchmarks/benchmark-100-v5/audit-reference-{reference_slug}.json")
    fresh_cases = root / "benchmarks/fresh-knowledge-v1/cases.json"
    fresh_candidate = root / "benchmarks/fresh-knowledge-v1/official-retrieval.json"
    fresh_reference = root / f"benchmarks/fresh-knowledge-v1/reference-{reference_slug}.json"
    candidate_score, total = _fresh_score(fresh_candidate)
    reference_score, reference_total = _fresh_score(fresh_reference)
    if total != reference_total:
        raise ValueError("fresh candidate and reference totals differ")
    comparisons = [
        {"name": "benchmark-100-v3", "suite_sha256": v3_candidate["suite_sha256"],
         "candidate": v3_candidate["correct"], "reference": v3_reference["correct"],
         "total": v3_candidate["rows"], "verified": v3_candidate["passed"] and v3_reference["passed"]},
        {"name": "benchmark-100-v5", "suite_sha256": v5_candidate["suite_sha256"],
         "candidate": v5_candidate["correct"], "reference": v5_reference["correct"],
         "total": v5_candidate["rows"], "verified": v5_candidate["passed"] and v5_reference["passed"]},
        {"name": "fresh-knowledge-v1", "suite_sha256": hashlib.sha256(fresh_cases.read_bytes()).hexdigest(),
         "candidate": candidate_score, "reference": reference_score, "total": total, "verified": True},
    ]
    return evaluate(comparisons, reference_parameters_b=reference_parameters_b, reference_name=reference_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify consistent LearnLoop 4B versus a larger reference")
    parser.add_argument("--reference-slug", default="qwen2.5-7b")
    parser.add_argument("--reference-name", default="Qwen2.5 7.6B")
    parser.add_argument("--reference-parameters-b", type=float, default=7.6)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/size-ladder-4b-vs-7.6b.json"))
    args = parser.parse_args()
    report = build(Path.cwd(), args.reference_slug, args.reference_name, args.reference_parameters_b)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
