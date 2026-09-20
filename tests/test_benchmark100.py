import inspect

from learnloop.benchmark100 import extract_answer, run_suite, spaced_sample


def test_extract_choice_uses_final_standalone_letter() -> None:
    assert extract_answer("I considered A, but the answer is C.", "choice") == "C"
    assert extract_answer("because", "choice") is None


def test_extract_number_uses_final_number() -> None:
    assert extract_answer("First 10, then final: 42", "number") == "42"
    assert extract_answer("1,234", "number") == "1234"


def test_spaced_sample_is_repeatable() -> None:
    rows = [{"n": number} for number in range(100)]
    assert spaced_sample(rows, 20, 7) == spaced_sample(rows, 20, 7)


def test_result_identity_includes_suite_hash_and_strategy() -> None:
    source = inspect.getsource(run_suite)
    assert 'suite_sha256[:12]' in source
    assert 'strategy-calc-v5-direct-choices' in source
    assert '"tool_trace": tool_trace' in source
