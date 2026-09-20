from fractions import Fraction

from learnloop.exact_solver import evaluate_arithmetic, solve_exact, solve_planned_expression


def test_safe_arithmetic_evaluator() -> None:
    assert evaluate_arithmetic("(16 - 3 - 4) * 2") == Fraction(18)
    assert evaluate_arithmetic("60 / 15 * 4") == Fraction(16)
    assert evaluate_arithmetic("140*6 + (140*0.9)*6") == Fraction(1596)


def test_safe_arithmetic_rejects_code_and_names() -> None:
    assert evaluate_arithmetic("__import__('os').system('echo nope')") is None
    assert evaluate_arithmetic("x + 1") is None
    assert evaluate_arithmetic("2 ** 100") is None


def test_planned_expression_requires_explicit_calc_tags() -> None:
    assert solve_planned_expression("<calc>(16 - 3 - 4) * 2</calc>") == "18"
    assert solve_planned_expression("The answer is 18") is None


def test_arithmetic() -> None:
    assert solve_exact("Compute (12 * 7) - (3 * 3). Give only the integer.") == 75


def test_recurrence() -> None:
    assert solve_exact("x0=2; x(n+1)=(3*x(n)+1) mod 997. Find x4. Give only the integer.") == 202


def test_grid() -> None:
    assert solve_exact("Count Right/Up paths from (0,0) to (2,2) avoiding [(1, 1)]. Give only the integer.") == 2


def test_subset_positions_are_distinct() -> None:
    assert solve_exact("Positions are distinct. How many subsets of [2, 2, 3] sum to 4? Give only the integer.") == 1


def test_unknown_problem_is_not_guessed() -> None:
    assert solve_exact("Explain photosynthesis.") is None
