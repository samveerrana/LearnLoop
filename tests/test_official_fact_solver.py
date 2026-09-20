from learnloop.official_fact_solver import solve_masked_year


def test_solves_masked_year_from_source_evidence() -> None:
    prompt = "Recall the four-digit year hidden in this learned fact: The system first flew in [BLANK]. Give only that year."
    assert solve_masked_year(prompt, "Background. The system first flew in 1966. Later changes followed.") == "1966"


def test_returns_none_when_evidence_does_not_match() -> None:
    prompt = "Recall the four-digit year hidden in this learned fact: The system first flew in [BLANK]. Give only that year."
    assert solve_masked_year(prompt, "A completely unrelated event happened in 1966.") is None
