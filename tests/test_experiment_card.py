import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads(ROOT.joinpath(relative).read_text(encoding="utf-8"))


def test_experiment_card_matches_authoritative_integrated_artifacts() -> None:
    base = load("benchmarks/integrated-v1-7b/summary-base.json")
    candidate = load("benchmarks/integrated-v1-7b/summary-candidate.json")
    reference = load("benchmarks/integrated-v1-7b/summary-reference.json")
    gate = load("benchmarks/integrated-v1-7b/v1-gate.json")
    card = ROOT.joinpath("docs/EXPERIMENT_CARD.md").read_text(encoding="utf-8")
    assert (base["correct"], candidate["correct"], reference["correct"]) == (70, 104, 69)
    assert base["total"] == candidate["total"] == reference["total"] == 159
    assert "70/159" in card and "104/159" in card and "69/159" in card
    failed = [name for name, passed in gate["checks"].items() if not passed]
    assert failed == ["real_approximately_20b_reference"]


def test_integrated_candidate_audit_observes_every_required_feature() -> None:
    audit = load("benchmarks/integrated-v1-7b/audit-candidate.json")
    assert audit["passed"]
    assert audit["observed_features"] == {
        "verified_tools": True,
        "temporary_parameter_rewrite": True,
        "official_retrieval": True,
    }
