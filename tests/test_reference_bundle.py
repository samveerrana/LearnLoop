import json

from learnloop.reference_bundle import BUNDLE_FILES, execution_safety, sha256, verify_bundle


def test_model_specific_execution_safety() -> None:
    manifest = {
        "minimum_physical_memory_gib": 48,
        "minimum_accelerator_memory_gib": 15,
        "minimum_host_memory_with_accelerator_gib": 12,
        "minimum_combined_memory_gib": 27,
        "minimum_free_disk_gib": 10,
    }
    assert not execution_safety(manifest, 20 * 1024**3, 16 * 1024**3)["safe"]
    assert execution_safety(manifest, 10 * 1024**3, 48 * 1024**3)["safe"]
    cloud = execution_safety(
        manifest, 10 * 1024**3, 12 * 1024**3, accelerator_bytes=15 * 1024**3,
    )
    assert cloud["safe"] and cloud["dedicated_accelerator_path"]


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
