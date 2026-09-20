import json
from pathlib import Path

from learnloop.benchmark_audit import audit


def write_case(tmp_path, *, duplicate=False, wrong_expected=False, wrong_category=False):
    suite = {"cases": [{"id": "Q1", "category": "x", "expected": "A"}]}
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite))
    import hashlib
    digest = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    row = {"id": "Q1", "category": "y" if wrong_category else "x", "expected": "B" if wrong_expected else "A", "answer": "A", "correct": True}
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(row) + "\n" + (json.dumps(row) + "\n" if duplicate else ""))
    summary = {"suite_sha256": digest, "results": str(results), "correct": 1, "total": 1, "categories": {"x": {"correct": 1, "total": 1}}}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    return suite_path, summary_path


def test_audit_accepts_consistent_complete_run(tmp_path) -> None:
    suite, summary = write_case(tmp_path)
    assert audit(suite, summary)["passed"]


def test_audit_rejects_duplicate_and_answer_tampering(tmp_path) -> None:
    suite, summary = write_case(tmp_path, duplicate=True, wrong_expected=True)
    report = audit(suite, summary)
    assert not report["passed"]
    assert any("duplicate" in error for error in report["errors"])
    assert any("expected answer" in error for error in report["errors"])


def test_audit_rejects_category_tampering(tmp_path) -> None:
    suite, summary = write_case(tmp_path, wrong_category=True)
    report = audit(suite, summary)
    assert not report["passed"]
    assert any("category differs" in error for error in report["errors"])


def test_audit_derives_feature_use_from_raw_trace(tmp_path) -> None:
    suite, summary = write_case(tmp_path)
    summary_data = json.loads(summary.read_text())
    results = summary_data["results"]
    results_path = Path(results)
    row = json.loads(results_path.read_text().splitlines()[0])
    row.update({
        "used": "temporary-parameter-rewrite",
        "training_trace": {"retrieval_api": "Wikipedia MediaWiki API", "scraping": False},
    })
    results_path.write_text(json.dumps(row) + "\n")
    observed = audit(suite, summary)["observed_features"]
    assert observed["temporary_parameter_rewrite"]
    assert observed["official_retrieval"]
    assert not observed["verified_tools"]
