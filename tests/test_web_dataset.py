import json

import pytest

from learnloop.web_dataset import build_licensed_web_dataset


def test_web_dataset_has_distinct_closed_book_prompts(tmp_path) -> None:
    content = " ".join([
        "The example project began in 1841 after a long public discussion.",
        "Its second documented phase started in 1872 and changed the design.",
        "A public exhibition opened in 1904 with several new demonstrations.",
        "The final historical restoration was completed in 2007 by local experts.",
    ])
    counts = build_licensed_web_dataset([{
        "title": "Example", "content": content,
        "source_url": "https://en.wikipedia.org/wiki/Example",
    }], tmp_path)
    assert counts["closed_book_test"] == 4
    train_text = tmp_path.joinpath("train.jsonl").read_text()
    tests = [json.loads(line) for line in tmp_path.joinpath("closed-book-test.jsonl").read_text().splitlines()]
    assert all(case["prompt"] not in train_text for case in tests)
    assert {case["expected"] for case in tests} == {"1841", "1872", "1904", "2007"}


def test_web_dataset_rejects_non_allowlisted_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        build_licensed_web_dataset([{
            "title": "No", "content": "A sufficiently long untrusted sentence mentions the year 1900. " * 5,
            "source_url": "https://example.com/no",
        }], tmp_path)
