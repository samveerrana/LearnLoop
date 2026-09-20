import json

from learnloop.reference_bundle import BUNDLE_FILES, sha256, verify_bundle


def make_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    suite = bundle / "suite.json"
    suite.write_text("{}")
    results = bundle / "core-results.jsonl"
    results.write_text("{}\n")
    (bundle / "reference-manifest.json").write_text(json.dumps({"model": "gpt-oss:20b"}))
    (bundle / "core-attestation.json").write_text(json.dumps({
        "model": "gpt-oss:20b", "suite_sha256": sha256(suite), "results_sha256": sha256(results),
    }))
    (bundle / "knowledge-results.json").write_text(json.dumps({"model": "gpt-oss:20b", "rows": []}))
    manifest = {
        "suite_sha256": sha256(suite),
        "sha256": {name: sha256(bundle / name) for name in BUNDLE_FILES},
    }
    (bundle / "bundle-manifest.json").write_text(json.dumps(manifest))
    return bundle


def test_verify_bundle_accepts_matching_hashes_and_identity(tmp_path) -> None:
    assert verify_bundle(make_bundle(tmp_path))["passed"]


def test_verify_bundle_rejects_changed_bytes(tmp_path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "core-results.jsonl").write_text("changed")
    report = verify_bundle(bundle)
    assert not report["passed"]
    assert "hash mismatch for core-results.jsonl" in report["errors"]
