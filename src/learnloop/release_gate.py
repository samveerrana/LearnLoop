from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_gate(
    base: dict,
    candidate: dict,
    reference: dict,
    reference_manifest: dict,
    feature_evidence: dict,
    audits: list[dict],
    minimum_reference_b: float = 18.0,
) -> dict:
    reference_parameters_b = float(reference_manifest.get("total_parameters_b", 0))
    base_categories = base.get("categories", {})
    candidate_categories = candidate.get("categories", {})
    regressions = {
        name: {
            "base": values["correct"],
            "candidate": candidate_categories.get(name, {}).get("correct", 0),
        }
        for name, values in base_categories.items()
        if candidate_categories.get(name, {}).get("correct", 0) < values["correct"]
    }
    required_features = ("official_non_scraping_retrieval", "temporary_parameter_rewrite", "verified_tools")
    missing_features = [name for name in required_features if not feature_evidence.get(name)]
    candidate_audit = audits[1] if len(audits) > 1 else {}
    invocation = candidate_audit.get("observed_features", {})
    missing_invocations = [
        name for name in ("official_retrieval", "temporary_parameter_rewrite", "verified_tools")
        if not invocation.get(name)
    ]
    suite_hashes = {base.get("suite_sha256"), candidate.get("suite_sha256"), reference.get("suite_sha256")}
    identical_frozen_suite = None not in suite_hashes and len(suite_hashes) == 1
    audit_hashes = {item.get("suite_sha256") for item in audits}
    audits_pass = len(audits) == 3 and all(item.get("passed") for item in audits) and audit_hashes == suite_hashes
    official_reference_provenance = all(
        str(reference_manifest.get(name, "")).startswith("https://")
        for name in ("official_model_page", "official_upstream_model_card")
    )
    checks = {
        "real_approximately_20b_reference": reference_parameters_b >= minimum_reference_b,
        "reference_identity_matches_manifest": reference.get("model") == reference_manifest.get("model"),
        "official_reference_provenance": official_reference_provenance,
        "all_raw_runs_independently_audited": audits_pass,
        "same_frozen_suite_hash_and_size": identical_frozen_suite and base.get("total") == candidate.get("total") == reference.get("total"),
        "candidate_matches_or_beats_reference": candidate.get("correct", 0) >= reference.get("correct", 0),
        "no_category_regressions_vs_base": not regressions,
        "all_framework_capabilities_verified": not missing_features,
        "all_required_features_invoked_in_candidate": not missing_invocations,
        "weight_rewrite_has_heldout_gain": feature_evidence.get("weight_rewrite_heldout_gain", 0) > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scores": {"base_4b": base.get("correct"), "candidate_4b": candidate.get("correct"), "reference": reference.get("correct")},
        "reference_parameters_b": reference_parameters_b,
        "reference_active_parameters_b": reference_manifest.get("active_parameters_b"),
        "reference_model": reference_manifest.get("model"),
        "minimum_reference_b": minimum_reference_b,
        "suite_sha256": base.get("suite_sha256") if identical_frozen_suite else None,
        "category_regressions": regressions,
        "missing_features": missing_features,
        "missing_benchmark_invocations": missing_invocations,
        "warning": "A benchmark score is task-scoped and is not a universal intelligence multiplier.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict LearnLoop V1 release gate")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    result = evaluate_gate(
        load(args.base), load(args.candidate), load(args.reference), load(args.reference_manifest),
        load(args.features), [load(args.base_audit), load(args.candidate_audit), load(args.reference_audit)],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
