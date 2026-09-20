from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .benchmark_audit import audit


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_rows(summary: dict) -> list[dict]:
    return [
        json.loads(line) for line in Path(summary["results"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized_core(rows: list[dict]) -> list[dict]:
    return [{**row, "id": f"CORE::{row['id']}"} for row in rows]


def normalized_knowledge(report: dict, role: str) -> list[dict]:
    rows = []
    for source in report["rows"]:
        row = {
            "id": f"KNOWLEDGE::{source['id']}",
            "category": "knowledge",
            "expected": source["expected"],
            "answer": source.get("answer"),
            "correct": source.get("answer") == source["expected"],
            "raw": source.get("raw", ""),
            "seconds": source.get("seconds"),
            "source_url": source["source_url"],
            "used": (
                "official-retrieval" if role == "candidate" and report.get("retrieval_during_evaluation")
                else "temporary-parameter-rewrite" if role == "candidate" else "model"
            ),
            "tool_trace": None,
        }
        if role == "candidate" and report.get("retrieval_during_evaluation"):
            row["training_trace"] = {
                "retrieval_api": "Wikipedia MediaWiki API", "scraping": False,
                "source_license": "CC-BY-SA-4.0", "retrieval_during_evaluation": True,
                "evidence_matcher": "masked fact with conservative lexical fallback",
            }
        elif role == "candidate":
            row["training_trace"] = {
                "retrieval_api": "Wikipedia MediaWiki API",
                "scraping": False,
                "source_license": "CC-BY-SA-4.0",
                "update": "rank-8 LoRA over all 36 layers; 16,515,072 trainable parameters",
                "temporary_capsule_deleted": True,
                "retrieval_during_evaluation": False,
            }
        rows.append(row)
    return rows


def make_summary(name: str, model: str, suite_hash: str, rows: list[dict], results_path: Path) -> dict:
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        categories[row["category"]]["total"] += 1
        categories[row["category"]]["correct"] += int(row["correct"])
    return {
        "suite": "LearnLoop Integrated Benchmark",
        "suite_sha256": suite_hash,
        "model": model,
        "backend": "composite-audited-local",
        "mode": name,
        "correct": sum(row["correct"] for row in rows),
        "total": len(rows),
        "categories": dict(sorted(categories.items())),
        "results": str(results_path),
    }


def compose_candidate_only(core_suite_path: Path, core_summary_path: Path, knowledge_path: Path, output_dir: Path) -> dict:
    core_suite, core_summary, knowledge = load(core_suite_path), load(core_summary_path), load(knowledge_path)
    expected_hash = hashlib.sha256(core_suite_path.read_bytes()).hexdigest()
    if core_summary.get("suite_sha256") != expected_hash:
        raise ValueError("core summary does not match frozen core suite")
    suite = {
        "name": "LearnLoop Fresh Integrated Candidate Benchmark",
        "count": len(core_suite["cases"]) + len(knowledge["rows"]),
        "core_suite_sha256": expected_hash,
        "sources": {"core": core_suite.get("sources", {}), "knowledge": "Wikipedia MediaWiki API; CC-BY-SA-4.0; no scraping"},
        "cases": [{**case, "id": f"CORE::{case['id']}"} for case in core_suite["cases"]] + [
            {"id": f"KNOWLEDGE::{row['id']}", "category": "knowledge", "kind": "year",
             "prompt": row["prompt"], "expected": row["expected"], "source_url": row["source_url"]}
            for row in knowledge["rows"]
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = output_dir / "suite.json"
    suite_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    suite_hash = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    rows = normalized_core(raw_rows(core_summary)) + normalized_knowledge(knowledge, "candidate")
    results_path = output_dir / "results-candidate.jsonl"
    results_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = make_summary("candidate", core_summary["model"], suite_hash, rows, results_path)
    summary_path = output_dir / "summary-candidate.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = audit(suite_path, summary_path)
    (output_dir / "audit-candidate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise ValueError(f"candidate composite failed audit: {report['errors']}")
    return {"suite_sha256": suite_hash, "score": summary["correct"], "total": summary["total"], "audit": report}


def compose(
    core_suite_path: Path,
    base_core_summary_path: Path,
    candidate_core_summary_path: Path,
    reference_core_summary_path: Path,
    base_knowledge_path: Path,
    candidate_knowledge_path: Path,
    reference_knowledge_path: Path,
    output_dir: Path,
) -> dict:
    core_suite = load(core_suite_path)
    base_core, candidate_core, reference_core = map(
        load, (base_core_summary_path, candidate_core_summary_path, reference_core_summary_path)
    )
    expected_hash = hashlib.sha256(core_suite_path.read_bytes()).hexdigest()
    for summary in (base_core, candidate_core, reference_core):
        if summary.get("suite_sha256") != expected_hash:
            raise ValueError("core summary does not match frozen core suite")
    base_knowledge, candidate_knowledge, reference_knowledge = map(
        load, (base_knowledge_path, candidate_knowledge_path, reference_knowledge_path)
    )
    knowledge_expected = {(row["id"], row["expected"]) for row in base_knowledge["rows"]}
    for report in (candidate_knowledge, reference_knowledge):
        if {(row["id"], row["expected"]) for row in report["rows"]} != knowledge_expected:
            raise ValueError("knowledge reports do not cover identical frozen cases")

    suite = {
        "name": "LearnLoop Integrated Benchmark",
        "count": len(core_suite["cases"]) + len(base_knowledge["rows"]),
        "core_suite_sha256": expected_hash,
        "sources": {
            "core": core_suite.get("sources", {}),
            "knowledge": "Wikipedia MediaWiki API; CC-BY-SA-4.0; no scraping",
        },
        "cases": [
            {**case, "id": f"CORE::{case['id']}"} for case in core_suite["cases"]
        ] + [
            {
                "id": f"KNOWLEDGE::{row['id']}", "category": "knowledge", "kind": "year",
                "prompt": row["prompt"], "expected": row["expected"], "source_url": row["source_url"],
            }
            for row in base_knowledge["rows"]
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = output_dir / "suite.json"
    suite_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    suite_hash = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    roles = {
        "base": (base_core, base_knowledge),
        "candidate": (candidate_core, candidate_knowledge),
        "reference": (reference_core, reference_knowledge),
    }
    artifacts = {"suite": str(suite_path), "suite_sha256": suite_hash}
    for role, (core_summary, knowledge_report) in roles.items():
        rows = normalized_core(raw_rows(core_summary)) + normalized_knowledge(knowledge_report, role)
        results_path = output_dir / f"results-{role}.jsonl"
        results_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        summary = make_summary(role, core_summary["model"], suite_hash, rows, results_path)
        summary_path = output_dir / f"summary-{role}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        audit_report = audit(suite_path, summary_path)
        audit_path = output_dir / f"audit-{role}.json"
        audit_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
        if not audit_report["passed"]:
            raise ValueError(f"{role} composite failed audit: {audit_report['errors']}")
        artifacts[role] = {"summary": str(summary_path), "audit": str(audit_path), "score": summary["correct"]}
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose audited core and temporary-rewrite benchmark runs")
    parser.add_argument("--core-suite", type=Path, required=True)
    parser.add_argument("--base-core", type=Path, required=True)
    parser.add_argument("--candidate-core", type=Path, required=True)
    parser.add_argument("--reference-core", type=Path, required=True)
    parser.add_argument("--base-knowledge", type=Path, required=True)
    parser.add_argument("--candidate-knowledge", type=Path, required=True)
    parser.add_argument("--reference-knowledge", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compose(
        args.core_suite, args.base_core, args.candidate_core, args.reference_core,
        args.base_knowledge, args.candidate_knowledge, args.reference_knowledge, args.output_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
