from learnloop.choice_tool import extract_choice, extract_leading_choice, majority_choice


def test_majority_choice_and_stable_tie_break() -> None:
    assert majority_choice(["B", "A", "B"]) == "B"
    assert majority_choice(["C", "A", "B"]) == "C"
    assert majority_choice([None, None]) is None


def test_extract_choice_uses_final_standalone_letter() -> None:
    assert extract_choice("A seems possible, but final answer: D") == "D"
    assert extract_choice("Choice is (C).") == "C"
    assert extract_choice("because") is None


def test_extract_choice_rejects_option_letters_in_truncated_reasoning() -> None:
    assert extract_choice("Option A looks plausible, but we still need to compare B and") is None


def test_extract_leading_choice_uses_only_answer_first_protocol() -> None:
    assert extract_leading_choice("B. Short check follows") == "B"
    assert extract_leading_choice("(C) because...") == "C"
    assert extract_leading_choice("I think B") is None
