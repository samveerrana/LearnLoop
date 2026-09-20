from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen

from .exact_solver import arithmetic_planner_prompt, solve_exact, solve_model_planned_arithmetic, solve_planned_expression
from .memory import MemoryStore
from .ollama_chat import ollama_answer, unload
from .research import WikipediaResearcher, should_research


API = "https://datasets-server.huggingface.co/rows"
LETTERS = "ABCD"
MMLU_CONFIGS = (
    "anatomy", "astronomy", "business_ethics", "college_biology", "college_chemistry",
    "college_mathematics", "conceptual_physics", "econometrics", "formal_logic", "global_facts",
    "high_school_geography", "high_school_psychology", "international_law", "machine_learning",
    "management", "medical_genetics", "philosophy", "prehistory", "sociology", "world_religions",
)
SOURCES = {
    "mmlu": {"dataset": "cais/mmlu", "config": "all", "split": "test", "license": "MIT"},
    "arc": {"dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test", "license": "CC-BY-SA-4.0"},
    "gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test", "license": "MIT"},
    "computer_science": {"dataset": "cais/mmlu", "config": "college_computer_science", "split": "test", "license": "MIT"},
}


def api_rows(source: dict, length: int = 100) -> list[dict]:
    query = urlencode({"dataset": source["dataset"], "config": source["config"], "split": source["split"], "offset": 0, "length": length})
    url = f"{API}?{query}"
    for attempt in range(6):
        try:
            with urlopen(url, timeout=60) as response:
                return [item["row"] for item in json.load(response)["rows"]]
        except HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 16)
            time.sleep(delay)
    raise RuntimeError("unreachable API retry state")


def spaced_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), count))
    return [rows[index] for index in indices]


def choices_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{LETTERS[index]}. {choice}" for index, choice in enumerate(choices))
    return f"{question}\n{options}\nAnswer with only A, B, C, or D."


def build_suite(
    root: Path,
    seed: int = 20260913,
    suite_dir: str = "benchmark-100",
    exclude_suites: Iterable[Path] | None = None,
) -> Path:
    destination = root / "benchmarks" / suite_dir
    destination.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    excluded_prompts = set()
    for exclude_suite in exclude_suites or ():
        excluded_prompts.update(
            case["prompt"] for case in json.loads(exclude_suite.read_text(encoding="utf-8"))["cases"]
        )

    for category in ("mmlu", "arc", "gsm8k", "computer_science"):
        if category == "mmlu":
            rows = []
            for config in MMLU_CONFIGS:
                source = {**SOURCES[category], "config": config}
                candidates = [
                    row for row in api_rows(source, 100)
                    if choices_prompt(row["question"], row["choices"]) not in excluded_prompts
                ]
                rows.append(candidates[random.Random(f"{seed}:{config}").randrange(len(candidates))])
        else:
            candidates = api_rows(SOURCES[category], 100)
            if category == "arc":
                candidates = [
                    row for row in candidates
                    if choices_prompt(row["question"], row["choices"]["text"]) not in excluded_prompts
                ]
            elif category == "gsm8k":
                candidates = [
                    row for row in candidates
                    if row["question"] + "\nGive only the final number." not in excluded_prompts
                ]
            else:
                candidates = [
                    row for row in candidates
                    if choices_prompt(row["question"], row["choices"]) not in excluded_prompts
                ]
            rows = spaced_sample(candidates, 20, seed + len(cases))
        for index, row in enumerate(rows):
            item_id = f"{category.upper()}-{index + 1:02d}"
            if category in {"mmlu", "computer_science"}:
                cases.append({"id": item_id, "category": category, "kind": "choice", "prompt": choices_prompt(row["question"], row["choices"]), "expected": LETTERS[row["answer"]]})
            elif category == "arc":
                cases.append({"id": item_id, "category": category, "kind": "choice", "prompt": choices_prompt(row["question"], row["choices"]["text"]), "expected": row["answerKey"]})
            else:
                expected = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", row["answer"])
                cases.append({"id": item_id, "category": category, "kind": "number", "prompt": row["question"] + "\nGive only the final number.", "expected": expected.group(1).replace(",", "")})

    quiz = root / "benchmarks" / "quiz-1000"
    questions = [json.loads(line) for line in quiz.joinpath("questions.jsonl").read_text().splitlines()]
    answers = {row["id"]: row["answer"] for row in map(json.loads, quiz.joinpath("private-answers.jsonl").read_text().splitlines())}
    tool_candidates = [row for row in questions if row["question"] not in excluded_prompts]
    for index, row in enumerate(spaced_sample(tool_candidates, 20, seed + 99)):
        cases.append({"id": f"TOOLS-{index + 1:02d}", "category": "tools", "kind": "number", "prompt": row["question"], "expected": str(answers[row["id"]])})

    suite = destination / "suite.json"
    suite.write_text(json.dumps({"name": f"LearnLoop Benchmark 100 ({suite_dir})", "seed": seed, "count": len(cases), "sources": SOURCES, "retrieval_method": "official Hugging Face Dataset Viewer API", "cases": cases}, indent=2), encoding="utf-8")
    return suite


def extract_answer(text: str, kind: str) -> str | None:
    if kind == "choice":
        matches = re.findall(r"(?<![A-Z])[ABCD](?![A-Z])", text.upper())
        return matches[-1] if matches else None
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def run_suite(root: Path, suite_path: Path, model: str, mode: str, limit: int | None = None, model_path: Path | None = None, adapter_path: Path | None = None, adapter_categories: set[str] | None = None, categories: set[str] | None = None, max_tokens: int = 64) -> dict:
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    suite_sha256 = hashlib.sha256(suite_bytes).hexdigest()
    cases = [case for case in suite["cases"] if not categories or case["category"] in categories]
    cases = cases[:limit]
    backend = "mlx" if model_path else "ollama"
    run_name = f"{model}-capsule-{'-'.join(sorted(adapter_categories or [])) or 'all'}" if adapter_path else model
    if mode == "learnloop":
        run_name += "-strategy-calc-v5-direct-choices"
    if max_tokens != 64:
        run_name += f"-tokens{max_tokens}"
    run_name += f"-suite-{suite_sha256[:12]}"
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name)
    output = suite_path.parent / f"results-{mode}-{safe_model}.jsonl"
    completed = {row["id"]: row for row in map(json.loads, output.read_text().splitlines())} if output.exists() else {}
    scratch = Path(tempfile.mkdtemp(prefix="learnloop-benchmark100-"))
    mlx_model = tokenizer = None
    loaded_adapter: Path | None | str = "unloaded"
    try:
        with output.open("a", encoding="utf-8") as handle:
            for position, case in enumerate(cases, 1):
                if case["id"] in completed:
                    continue
                started = time.monotonic()
                raw = ""
                used = "model"
                tool_trace = None
                if mode == "learnloop" and case["category"] == "tools":
                    exact = solve_exact(case["prompt"])
                    if exact is not None:
                        raw, used = str(exact), "verified-local-tool"
                if mode == "learnloop" and case["category"] == "gsm8k" and model_path:
                    wanted_adapter = adapter_path if adapter_path and (not adapter_categories or case["category"] in adapter_categories) else None
                    if loaded_adapter != wanted_adapter:
                        if mlx_model is not None:
                            del mlx_model
                            gc.collect()
                            import mlx.core as mx
                            mx.clear_cache()
                        from mlx_lm import load
                        mlx_model, tokenizer = load(str(model_path), adapter_path=str(wanted_adapter) if wanted_adapter else None)
                        loaded_adapter = wanted_adapter
                    planned = solve_model_planned_arithmetic(mlx_model, tokenizer, case["prompt"])
                    if planned is not None:
                        raw, used = planned, "model-planned-verified-calculator"
                if mode == "learnloop" and case["category"] == "gsm8k" and not model_path:
                    planner_raw = ollama_answer(
                        model, arithmetic_planner_prompt(case["prompt"]),
                        MemoryStore(scratch / f"{case['id']}-planner.sqlite3"), enhanced=False,
                        thinking=False, max_tokens=192,
                        system_prompt="You produce safe arithmetic expressions for a verified calculator.",
                    )
                    tool_trace = {"planner_output": planner_raw}
                    planned = solve_planned_expression(planner_raw)
                    if planned is not None:
                        raw, used = planned, "model-planned-verified-calculator"
                if not raw:
                    memory = MemoryStore(scratch / f"{case['id']}.sqlite3")
                    strict = (
                        "Solve the problem privately. Return only the requested final letter or number, "
                        "with no explanation, citations, or extra words."
                    )
                    if model_path:
                        wanted_adapter = adapter_path if adapter_path and (not adapter_categories or case["category"] in adapter_categories) else None
                        if loaded_adapter != wanted_adapter:
                            if mlx_model is not None:
                                del mlx_model
                                gc.collect()
                                import mlx.core as mx
                                mx.clear_cache()
                            from mlx_lm import load
                            mlx_model, tokenizer = load(str(model_path), adapter_path=str(wanted_adapter) if wanted_adapter else None)
                            loaded_adapter = wanted_adapter
                        evidence = ""
                        if mode == "learnloop" and should_research(case["prompt"]):
                            try:
                                WikipediaResearcher(memory).research(case["prompt"])
                                evidence = "\n".join(item["content"][:1800] for item in memory.search_knowledge(case["prompt"]))
                            except Exception:
                                pass
                        user = (f"Reference evidence (use only if relevant):\n{evidence}\n\n" if evidence else "") + case["prompt"]
                        prompt = tokenizer.apply_chat_template([{"role": "system", "content": strict}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
                        from mlx_lm import generate
                        raw = generate(mlx_model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False).strip()
                    else:
                        raw = ollama_answer(
                            model,
                            case["prompt"],
                            memory,
                            enhanced=mode == "learnloop",
                            thinking=False,
                            max_tokens=max_tokens,
                            system_prompt=strict,
                        )
                answer = extract_answer(raw, case["kind"])
                row = {"id": case["id"], "category": case["category"], "expected": case["expected"], "answer": answer, "correct": answer == case["expected"], "used": used, "tool_trace": tool_trace, "seconds": round(time.monotonic() - started, 2), "raw": raw}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                completed[case["id"]] = row
                print(f"{mode} {position}/{len(cases)} {case['id']}: {'PASS' if row['correct'] else 'FAIL'}", flush=True)
    finally:
        if not model_path:
            try:
                unload(model)
            except Exception:
                pass

    rows = [completed[case["id"]] for case in cases if case["id"] in completed]
    categories = {category: {"correct": sum(row["correct"] for row in rows if row["category"] == category), "total": sum(row["category"] == category for row in rows)} for category in sorted({row["category"] for row in rows})}
    summary = {"suite": suite["name"], "suite_sha256": suite_sha256, "model": model, "backend": backend, "adapter": str(adapter_path) if adapter_path else None, "adapter_categories": sorted(adapter_categories or []), "mode": mode, "correct": sum(row["correct"] for row in rows), "total": len(rows), "categories": categories, "results": str(output)}
    (suite_path.parent / f"summary-{mode}-{safe_model}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or run the transparent LearnLoop Benchmark 100")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--mode", choices=("base", "reference", "learnloop"), default="base")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-path", type=Path, help="Use an MLX checkpoint instead of Ollama")
    parser.add_argument("--adapter-path", type=Path, help="Temporary MLX parameter capsule")
    parser.add_argument("--adapter-categories", help="Comma-separated categories that may load the capsule")
    parser.add_argument("--suite-dir", default="benchmark-100")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument(
        "--exclude-suite", type=Path, action="append", default=[],
        help="Exclude every prompt from an earlier frozen suite; may be repeated",
    )
    parser.add_argument("--categories", help="Evaluate only these comma-separated categories")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    root = Path.cwd()
    suite = root / "benchmarks" / args.suite_dir / "suite.json"
    if args.build or not suite.exists():
        suite = build_suite(root, args.seed, args.suite_dir, args.exclude_suite)
        print(f"Built {suite}")
    if not args.build:
        adapter_categories = set(args.adapter_categories.split(",")) if args.adapter_categories else None
        selected_categories = set(args.categories.split(",")) if args.categories else None
        run_suite(root, suite, args.model, args.mode, args.limit, args.model_path, args.adapter_path, adapter_categories, selected_categories, args.max_tokens)


if __name__ == "__main__":
    main()
