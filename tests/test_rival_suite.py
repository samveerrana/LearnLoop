from learnloop.rival_suite import correct


def test_short_answers_need_token_boundaries() -> None:
    assert correct("C", ["c"])
    assert correct("The answer is C.", ["c"])
    assert not correct("Because A is tallest", ["c"])
    assert not correct("not enough information", ["no"])


def test_long_answers_can_appear_in_explanation() -> None:
    assert correct("The command is git diff.", ["git diff"])
    assert correct("Water is H₂O.", ["h2o"])
    assert correct(r"The answer is \(\frac{2}{5}\).", ["2/5"])
