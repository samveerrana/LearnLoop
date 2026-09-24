from learnloop.release_gate import evaluate_gate


def summary(score: int) -> dict:
    return {"suite": "suite", "suite_sha256": "abc123", "model": "gpt-oss:20b", "total": 100, "correct": score, "categories": {"knowledge": {"correct": score, "total": 100}}}


def features() -> dict:
    return {
        "official_non_scraping_retrieval": True,
        "temporary_parameter_rewrite": True,
        "verified_tools": True,
        "weight_rewrite_heldout_gain": 1,
        "benchmark_feature_invocation": {
            "official_retrieval": True,
            "temporary_parameter_rewrite": True,
            "verified_tools": True,
        },
    }


def manifest(parameters=21) -> dict:
    return {"model": "gpt-oss:20b", "total_parameters_b": parameters, "active_parameters_b": 3.6, "official_model_page": "https://ollama.com/model", "official_upstream_model_card": "https://huggingface.co/model"}


def audits() -> list[dict]:
    values = [{"passed": True, "suite_sha256": "abc123"} for _ in range(3)]
    values[1]["observed_features"] = {
        "official_retrieval": True,
        "temporary_parameter_rewrite": True,
        "verified_tools": True,
    }
    return values


def test_rejects_small_reference_even_when_score_wins() -> None:
    result = evaluate_gate(summary(50), summary(80), summary(60), manifest(7), features(), audits())
    assert not result["passed"]
    assert not result["checks"]["reference_meets_parameter_threshold"]


def test_30b_gate_rejects_21b_reference() -> None:
    result = evaluate_gate(
        summary(50), summary(80), summary(60), manifest(21), features(), audits(),
        minimum_reference_b=30,
    )
    assert not result["passed"]
    assert not result["checks"]["reference_meets_parameter_threshold"]


def test_30b_gate_accepts_complete_32b_evidence() -> None:
    reference_manifest = manifest(32.8)
    assert evaluate_gate(
        summary(50), summary(80), summary(75), reference_manifest, features(), audits(),
        minimum_reference_b=30,
    )["passed"]


def test_rejects_category_regression() -> None:
    result = evaluate_gate(summary(70), summary(69), summary(60), manifest(), features(), audits())
    assert not result["passed"]
    assert result["category_regressions"]


def test_passes_only_complete_evidence() -> None:
    assert evaluate_gate(summary(50), summary(80), summary(75), manifest(), features(), audits())["passed"]


def test_rejects_different_or_missing_suite_hash() -> None:
    candidate = summary(80)
    candidate["suite_sha256"] = "different"
    result = evaluate_gate(summary(50), candidate, summary(75), manifest(), features(), audits())
    assert not result["checks"]["same_frozen_suite_hash_and_size"]

    missing = summary(80)
    del missing["suite_sha256"]
    result = evaluate_gate(summary(50), missing, summary(75), manifest(), features(), audits())
    assert not result["checks"]["same_frozen_suite_hash_and_size"]


def test_rejects_spoofed_identity_or_missing_audit() -> None:
    wrong = summary(75)
    wrong["model"] = "small-model:7b"
    result = evaluate_gate(summary(50), summary(80), wrong, manifest(), features(), audits())
    assert not result["checks"]["reference_identity_matches_manifest"]
    result = evaluate_gate(summary(50), summary(80), summary(75), manifest(), features(), audits()[:2])
    assert not result["checks"]["all_raw_runs_independently_audited"]


def test_rejects_capability_that_was_not_invoked_in_candidate_run() -> None:
    run_audits = audits()
    run_audits[1]["observed_features"]["temporary_parameter_rewrite"] = False
    result = evaluate_gate(summary(50), summary(80), summary(75), manifest(), features(), run_audits)
    assert not result["passed"]
    assert not result["checks"]["all_required_features_invoked_in_candidate"]
    assert result["missing_benchmark_invocations"] == ["temporary_parameter_rewrite"]
