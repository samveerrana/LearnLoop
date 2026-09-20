from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .reference_preflight import assess, physical_memory_bytes


BUNDLE_FILES = (
    "suite.json",
    "core-results.jsonl",
    "core-attestation.json",
    "knowledge-results.json",
    "reference-manifest.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bundle(bundle: Path) -> dict:
    manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for name in BUNDLE_FILES:
        path = bundle / name
        if not path.is_file():
            errors.append(f"missing {name}")
        elif manifest.get("sha256", {}).get(name) != sha256(path):
            errors.append(f"hash mismatch for {name}")
    if not errors:
        suite_hash = sha256(bundle / "suite.json")
        if manifest.get("suite_sha256") != suite_hash:
            errors.append("bundle suite hash mismatch")
        reference = json.loads((bundle / "reference-manifest.json").read_text(encoding="utf-8"))
        attestation = json.loads((bundle / "core-attestation.json").read_text(encoding="utf-8"))
        knowledge = json.loads((bundle / "knowledge-results.json").read_text(encoding="utf-8"))
        if attestation.get("model") != reference.get("model") or knowledge.get("model") != reference.get("model"):
            errors.append("model identity differs across bundled artifacts")
        if attestation.get("suite_sha256") != suite_hash:
            errors.append("core attestation suite hash mismatch")
        if attestation.get("results_sha256") != sha256(bundle / "core-results.jsonl"):
            errors.append("core result hash differs from attestation")
    return {"passed": not errors, "bundle": str(bundle), "errors": errors}


def create_bundle(suite: Path, knowledge_cases: Path, reference_manifest: Path, output: Path) -> dict:
    safety = assess(shutil.disk_usage(output.parent).free, physical_bytes=physical_memory_bytes())
    if not safety["safe_to_run_locally"]:
        raise RuntimeError(
            f"refusing reference job on {safety['physical_memory_bytes'] / 1024**3:.1f} GiB RAM; 24 GiB required"
        )
    manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
    model = manifest["model"]
    subprocess.run([
        sys.executable, "-m", "learnloop.reference_run", "--suite", str(suite),
        "--manifest", str(reference_manifest),
    ], check=True)
    suite_hash = sha256(suite)
    safe_model = "".join(character if character.isalnum() or character in "_.-" else "-" for character in model)
    results = suite.parent / f"results-reference-{safe_model}-suite-{suite_hash[:12]}.jsonl"
    attestation = suite.parent / f"attestation-reference-{safe_model}.json"
    knowledge_output = suite.parent / f"knowledge-reference-{safe_model}.json"
    subprocess.run([
        sys.executable, "-m", "learnloop.reference_knowledge_eval", "--model", model,
        "--cases", str(knowledge_cases), "--output", str(knowledge_output),
    ], check=True)
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        "suite.json": suite,
        "core-results.jsonl": results,
        "core-attestation.json": attestation,
        "knowledge-results.json": knowledge_output,
        "reference-manifest.json": reference_manifest,
    }
    for name, source in sources.items():
        shutil.copy2(source, output / name)
    bundle_manifest = {
        "schema_version": 1,
        "model": model,
        "suite_sha256": suite_hash,
        "sha256": {name: sha256(output / name) for name in BUNDLE_FILES},
        "instructions": "Copy this directory to the LearnLoop Mac and run learnloop-reference-bundle --verify BUNDLE.",
    }
    (output / "bundle-manifest.json").write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")
    report = verify_bundle(output)
    if not report["passed"]:
        raise RuntimeError(f"created bundle failed self-verification: {report['errors']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or verify a portable LearnLoop large-reference bundle")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", type=Path, metavar="OUTPUT_DIR")
    group.add_argument("--verify", type=Path, metavar="BUNDLE_DIR")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--knowledge-cases", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    args = parser.parse_args()
    if args.verify:
        report = verify_bundle(args.verify)
    else:
        if not all((args.suite, args.knowledge_cases, args.reference_manifest)):
            parser.error("--create requires --suite, --knowledge-cases, and --reference-manifest")
        try:
            report = create_bundle(args.suite, args.knowledge_cases, args.reference_manifest, args.create)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
