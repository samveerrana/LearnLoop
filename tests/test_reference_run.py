from learnloop.reference_run import safe_name


def test_reference_filename_is_deterministic_and_safe() -> None:
    assert safe_name("gpt-oss:20b") == "gpt-oss-20b"
