from learnloop.source_policy import training_permission


def test_wikipedia_is_allowlisted_with_license() -> None:
    result = training_permission("https://en.wikipedia.org/wiki/History_of_the_United_States")
    assert result.allowed
    assert result.license_id == "CC-BY-SA-4.0"


def test_unknown_web_source_is_not_automatically_trained() -> None:
    result = training_permission("https://example.com/article")
    assert not result.allowed
    assert result.license_id is None
