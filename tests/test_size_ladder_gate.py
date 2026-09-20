from learnloop.size_ladder_gate import evaluate


def test_requires_every_distinct_comparison_to_beat_reference() -> None:
    rows = [
        {"suite_sha256": char * 64, "candidate": candidate, "reference": reference, "verified": True}
        for char, candidate, reference in (("a", 70, 48), ("b", 75, 42), ("c", 54, 19))
    ]
    assert evaluate(rows)["passed"]
    rows[-1]["candidate"] = 18
    assert not evaluate(rows)["passed"]
