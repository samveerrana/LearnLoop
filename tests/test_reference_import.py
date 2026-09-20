import hashlib
import json

import pytest

from learnloop.reference_import import import_reference


def write_fixture(tmp_path):
    suite = {"name": "tiny", "cases": [
        {"id": "A", "category": "facts", "expected": "B"},
        {"id": "B", "category": "math", "expected": "4"},
    ]}
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite))
    rows = [
        {"id": "A", "category": "facts", "expected": "B", "answer": "B", "correct": True},
        {"id": "B", "category": "math", "expected": "4", "answer": "3", "correct": False},
    ]
    results_path = tmp_path / "results.jsonl"
    results_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"model": "gpt-oss:20b"}))
    attestation = {
        "model": "gpt-oss:20b",
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "ollama_show_sha256": "abc",
        "physical_memory_bytes": 32 * 1024**3,
        "platform": "test",
        "completed_at_utc": "2026-09-14T00:00:00+00:00",
        "command": "python -m learnloop.reference_run",
    }
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation))
    return suite_path, results_path, attestation_path, manifest_path


def test_import_reference_rebuilds_summary_and_passes_local_audit(tmp_path) -> None:
    inputs = write_fixture(tmp_path)
    report = import_reference(*inputs, tmp_path / "imported")
    assert report["audit"]["passed"]
    assert report["summary"]["correct"] == 1
    assert report["summary"]["total"] == 2
    assert report["summary"]["backend"] == "ollama-remote-import"


def test_import_reference_rejects_changed_result_bytes(tmp_path) -> None:
    suite, results, attestation, manifest = write_fixture(tmp_path)
    results.write_text(results.read_text() + "\n")
    with pytest.raises(ValueError, match="raw result bytes"):
        import_reference(suite, results, attestation, manifest, tmp_path / "imported")
