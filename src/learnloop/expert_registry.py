from __future__ import annotations

import json
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


MAX_STORED_PARAMETERS = 400_000_000
MAX_ACTIVE_PARAMETERS = 64_000_000
MANIFEST_NAME = "learnloop-expert.json"


def count_capsule_parameters(capsule: Path) -> int:
    """Count stored adapter tensor elements without loading them into accelerator memory."""
    from safetensors import safe_open

    total = 0
    files = sorted(capsule.glob("*.safetensors"))
    if not files:
        raise ValueError("capsule has no safetensors weight file")
    for path in files:
        with safe_open(path, framework="numpy") as weights:
            for key in weights.keys():
                total += _shape_size(weights.get_slice(key).get_shape())
    return total


def _shape_size(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


@dataclass(frozen=True)
class Expert:
    name: str
    path: Path
    base_model: str
    base_model_sha256: str
    parameters: int
    topics: tuple[str, ...]
    accepted: bool
    score_gain: float


class ExpertRegistry:
    """Disk-backed, reversible expert packs with strict parameter budgets."""

    def __init__(
        self,
        root: Path,
        stored_parameter_limit: int = MAX_STORED_PARAMETERS,
        active_parameter_limit: int = MAX_ACTIVE_PARAMETERS,
    ) -> None:
        self.root = root
        self.stored_parameter_limit = stored_parameter_limit
        self.active_parameter_limit = active_parameter_limit
        root.mkdir(parents=True, exist_ok=True)

    def experts(self) -> list[Expert]:
        found: list[Expert] = []
        for manifest_path in sorted(self.root.glob(f"*/{MANIFEST_NAME}")):
            found.append(self._read_expert(manifest_path))
        return found

    def _read_expert(self, manifest_path: Path) -> Expert:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_parameters = count_capsule_parameters(manifest_path.parent)
        if int(data["parameters"]) != actual_parameters:
            raise ValueError(f"expert parameter manifest mismatch: {manifest_path.parent.name}")
        if str(data["name"]) != manifest_path.parent.name:
            raise ValueError(f"expert name manifest mismatch: {manifest_path.parent.name}")
        return Expert(
            name=str(data["name"]), path=manifest_path.parent,
            base_model=str(data["base_model"]), base_model_sha256=str(data["base_model_sha256"]), parameters=actual_parameters,
            topics=tuple(str(topic).lower() for topic in data.get("topics", [])),
            accepted=bool(data.get("accepted", False)), score_gain=float(data.get("score_gain", 0)),
        )

    def stored_parameters(self) -> int:
        return sum(expert.parameters for expert in self.experts())

    def register(
        self,
        capsule: Path,
        *,
        name: str,
        topics: list[str],
        evaluation: dict,
    ) -> Expert:
        required = {"passed", "held_out", "suite_sha256", "base_model", "base_model_sha256", "base_score", "expert_score"}
        if not required <= evaluation.keys():
            raise ValueError("evaluation evidence is missing required fields")
        if evaluation["passed"] is not True or evaluation["held_out"] is not True:
            raise ValueError("expert rejected: evaluation is not a passing held-out audit")
        for field in ("suite_sha256", "base_model_sha256"):
            value = str(evaluation[field])
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                raise ValueError(f"expert rejected: invalid {field}")
        base_model = str(evaluation["base_model"])
        base_score, expert_score = int(evaluation["base_score"]), int(evaluation["expert_score"])
        if expert_score <= base_score:
            raise ValueError("expert rejected: held-out score did not improve")
        if capsule.is_symlink() or not capsule.is_dir():
            raise ValueError("capsule must be a directory")
        if any(path.is_symlink() for path in capsule.rglob("*")):
            raise ValueError("capsule must not contain symbolic links")
        parameters = count_capsule_parameters(capsule)
        if parameters <= 0:
            raise ValueError("expert parameter count must be positive")
        if self.stored_parameters() + parameters > self.stored_parameter_limit:
            raise ValueError("expert rejected: 400M stored-parameter budget would be exceeded")
        safe_name = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")
        if not safe_name:
            raise ValueError("expert name must contain letters or numbers")
        destination = self.root / safe_name
        if destination.exists():
            raise FileExistsError(f"expert already exists: {safe_name}")
        shutil.copytree(capsule, destination)
        manifest = {
            "name": safe_name,
            "base_model": base_model,
            "base_model_sha256": str(evaluation["base_model_sha256"]).lower(),
            "suite_sha256": str(evaluation["suite_sha256"]).lower(),
            "parameters": parameters,
            "topics": sorted({topic.strip().lower() for topic in topics if topic.strip()}),
            "base_score": base_score,
            "expert_score": expert_score,
            "score_gain": expert_score - base_score,
            "accepted": True,
            "temporary": True,
        }
        destination.joinpath(MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return self._read_expert(destination / MANIFEST_NAME)

    def route(self, question: str, base_model: str, base_model_sha256: str | None = None) -> list[Expert]:
        words = set(re.findall(r"[a-z0-9]+", question.lower()))
        ranked: list[tuple[int, float, Expert]] = []
        for expert in self.experts():
            if not expert.accepted or expert.base_model != base_model:
                continue
            if base_model_sha256 is not None and expert.base_model_sha256 != base_model_sha256.lower():
                continue
            topic_words = set(re.findall(r"[a-z0-9]+", " ".join(expert.topics)))
            overlap = len(words & topic_words)
            if overlap:
                ranked.append((overlap, expert.score_gain, expert))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected: list[Expert] = []
        active = 0
        for _, _, expert in ranked:
            if active + expert.parameters <= self.active_parameter_limit:
                selected.append(expert)
                active += expert.parameters
        return selected

    def delete(self, name: str) -> None:
        destination = (self.root / name).resolve()
        if destination.parent != self.root.resolve():
            raise ValueError("refusing to delete outside the expert registry")
        if destination.exists():
            shutil.rmtree(destination)


def hash_model(path: Path) -> str:
    """Hash model identity deterministically without loading model tensors into RAM."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())
    if not files:
        raise ValueError("model path contains no files")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
