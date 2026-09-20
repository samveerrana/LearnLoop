from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://en.wikipedia.org/w/api.php"
YEAR = r"(1[0-9]{3}|20[0-9]{2})"


def fetch_extract(title: str) -> str:
    query = urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                       "titles": title, "format": "json", "redirects": 1})
    request = Request(f"{API}?{query}", headers={"User-Agent": "LearnLoop/0.5 official-api-evaluation"})
    with urlopen(request, timeout=30) as response:
        pages = json.load(response).get("query", {}).get("pages", {})
    return next((page.get("extract", "") for page in pages.values()), "")


def masked_fact(prompt: str) -> str:
    marker = prompt.find(": ")
    end = prompt.rfind(" Give only that year.")
    return prompt[marker + 2:end] if marker >= 0 and end > marker else prompt


def solve_masked_year(prompt: str, source_text: str) -> str | None:
    fact = masked_fact(prompt)
    escaped = re.escape(fact).replace(r"\[BLANK\]", YEAR)
    escaped = re.sub(r"(?:\\ |\\\n)+", r"\\s+", escaped)
    match = re.search(escaped, source_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    # Robust fallback: compare source windows after removing years and punctuation.
    target = re.sub(r"\[BLANK\]", " ", fact)
    tokens = re.findall(r"[a-z0-9]+", target.lower())
    for year_match in re.finditer(YEAR, source_text):
        window = source_text[max(0, year_match.start() - 280):year_match.end() + 280]
        source_tokens = set(re.findall(r"[a-z0-9]+", window.lower()))
        if tokens and sum(token in source_tokens for token in tokens) / len(tokens) >= 0.90:
            return year_match.group(1)
    return None


def evaluate(input_report: Path, output: Path) -> dict:
    report = json.loads(input_report.read_text(encoding="utf-8"))
    cache = {title: fetch_extract(title) for title in sorted({row["title"] for row in report["rows"]})}
    rows = []
    for row in report["rows"]:
        answer = solve_masked_year(row["prompt"], cache[row["title"]])
        rows.append({**row, "answer": answer, "correct": answer == row["expected"],
                     "raw": answer, "used": "official-retrieval",
                     "training_trace": {"retrieval_api": "Wikipedia MediaWiki API", "scraping": False}})
    result = {"evaluation": "official-API retrieval with deterministic evidence matching",
              "retrieval_during_evaluation": True, "correct": sum(row["correct"] for row in rows),
              "total": len(rows), "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official-API factual retrieval without scraping")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.input, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
