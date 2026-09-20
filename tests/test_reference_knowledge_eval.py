from learnloop.reference_knowledge_eval import extract_year


def test_extract_year_uses_last_valid_four_digit_year() -> None:
    assert extract_year("Maybe 1968. Final answer: 1969") == "1969"
    assert extract_year("unknown") is None
