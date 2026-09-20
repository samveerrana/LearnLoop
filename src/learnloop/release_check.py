from __future__ import annotations

import json
from pathlib import Path


REQUIRED = ("README.md", "LICENSE", ".gitignore", "PUBLIC_V1_CHECKLIST.md", "pyproject.toml")
PRIVATE_PATTERNS = ("*.sqlite3", "*.safetensors", "*.gguf", "*.pem", "*.key", ".env")


def check(root: Path) -> dict:
    errors: list[str] = []
    for name in REQUIRED:
        if not root.joinpath(name).is_file():
            errors.append(f"missing required public file: {name}")
    for pattern in PRIVATE_PATTERNS:
        for path in root.glob(pattern):
            errors.append(f"private/generated root file must not be published: {path.name}")
    public_text = [root / "README.md", root / "PUBLIC_V1_CHECKLIST.md"]
    public_text.extend((root / "benchmarks/integrated-v1-7b").glob("*.json"))
    for path in public_text:
        if path.is_file() and "/Users/" in path.read_text(encoding="utf-8", errors="ignore"):
            errors.append(f"absolute home path exposed: {path}")
    gate = root / "benchmarks/integrated-v1-7b/v1-gate.json"
    if gate.is_file() and json.loads(gate.read_text()).get("passed") is True:
        errors.append("V1 gate unexpectedly claims completion without eligible 20B evidence")
    return {"passed": not errors, "errors": errors}


def main() -> None:
    root = Path.cwd()
    result = check(root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
