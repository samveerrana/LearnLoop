import json
from pathlib import Path

import pytest
import numpy as np
from safetensors.numpy import save_file

from learnloop.expert_registry import ExpertRegistry, _shape_size, hash_model


def evidence(base: int = 1, expert: int = 2, model: str = "qwen") -> dict:
    return {"passed": True, "held_out": True, "suite_sha256": "a" * 64,
            "base_model": model, "base_model_sha256": "b" * 64,
            "base_score": base, "expert_score": expert}


def capsule(path: Path, parameters: int = 40) -> Path:
    path.mkdir()
    save_file({"adapter": np.zeros((parameters,), dtype=np.float16)}, path / "adapters.safetensors")
    return path


def test_register_route_and_delete_reversible_expert(tmp_path: Path) -> None:
    registry = ExpertRegistry(tmp_path / "experts", stored_parameter_limit=100, active_parameter_limit=50)
    expert = registry.register(
        capsule(tmp_path / "candidate"), name="US History",
        topics=["United States history", "presidents"], evaluation=evidence(4, 7, "qwen-4b"),
    )
    assert registry.stored_parameters() == 40
    assert [item.name for item in registry.route("Explain United States history", "qwen-4b")] == ["us-history"]
    assert registry.route("Write Python", "qwen-4b") == []
    assert registry.route("United States history", "another-model") == []
    registry.delete(expert.name)
    assert registry.experts() == []


def test_rejects_non_improving_or_over_budget_expert(tmp_path: Path) -> None:
    registry = ExpertRegistry(tmp_path / "experts", stored_parameter_limit=30)
    candidate = capsule(tmp_path / "candidate")
    with pytest.raises(ValueError, match="did not improve"):
        registry.register(candidate, name="bad", topics=["x"], evaluation=evidence(2, 2))
    with pytest.raises(ValueError, match="400M"):
        registry.register(candidate, name="large", topics=["x"], evaluation=evidence(2, 3))


def test_router_respects_active_parameter_limit(tmp_path: Path) -> None:
    registry = ExpertRegistry(tmp_path / "experts", stored_parameter_limit=100, active_parameter_limit=45)
    for name, gain in (("history-a", 2), ("history-b", 1)):
        source = capsule(tmp_path / f"candidate-{name}", parameters=30)
        registry.register(source, name=name, topics=["history"], evaluation=evidence(1, 1 + gain))
    selected = registry.route("history", "qwen")
    assert len(selected) == 1
    assert selected[0].name == "history-a"


def test_manifest_is_inspectable(tmp_path: Path) -> None:
    registry = ExpertRegistry(tmp_path / "experts")
    expert = registry.register(
        capsule(tmp_path / "candidate", parameters=10), name="science",
        topics=["physics"], evaluation=evidence(),
    )
    manifest = json.loads(expert.path.joinpath("learnloop-expert.json").read_text())
    assert manifest["temporary"] is True
    assert manifest["accepted"] is True


def test_shape_size_counts_all_tensor_elements() -> None:
    assert _shape_size([2, 3, 4]) == 24


def test_tampered_parameter_manifest_fails_closed(tmp_path: Path) -> None:
    registry = ExpertRegistry(tmp_path / "experts")
    expert = registry.register(capsule(tmp_path / "candidate", 12), name="safe", topics=["x"], evaluation=evidence())
    manifest_path = expert.path / "learnloop-expert.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parameter manifest mismatch"):
        registry.experts()


def test_register_returns_new_expert_not_lexicographic_last(tmp_path: Path) -> None:
    registry = ExpertRegistry(tmp_path / "experts")
    registry.register(capsule(tmp_path / "z", 2), name="z-last", topics=["z"], evaluation=evidence())
    created = registry.register(capsule(tmp_path / "a", 2), name="a-first", topics=["a"], evaluation=evidence())
    assert created.name == "a-first"


def test_rejects_capsule_symlinks(tmp_path: Path) -> None:
    candidate = capsule(tmp_path / "candidate", 2)
    candidate.joinpath("outside-link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symbolic links"):
        ExpertRegistry(tmp_path / "experts").register(candidate, name="x", topics=["x"], evaluation=evidence())


def test_model_hash_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    model.joinpath("config.json").write_text("one")
    first = hash_model(model)
    assert hash_model(model) == first
    model.joinpath("config.json").write_text("two")
    assert hash_model(model) != first


def test_rejects_unverified_evaluation(tmp_path: Path) -> None:
    report = evidence()
    report["held_out"] = False
    with pytest.raises(ValueError, match="passing held-out audit"):
        ExpertRegistry(tmp_path / "experts").register(capsule(tmp_path / "candidate"), name="x", topics=["x"], evaluation=report)
