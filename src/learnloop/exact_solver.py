from __future__ import annotations

import ast
import re
from fractions import Fraction


ALLOWED_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def evaluate_arithmetic(expression: str) -> Fraction | None:
    """Evaluate only numeric +, -, *, / and parentheses; never execute arbitrary code."""
    if len(expression) > 300:
        return None
    try:
        root = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY:
            return ALLOWED_BINARY[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("unsafe expression")

    try:
        return visit(root)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def arithmetic_planner_prompt(question: str) -> str:
    return (
        "Translate this arithmetic word problem into one complete numeric expression. "
        "Use only decimal numbers, parentheses, +, -, *, and /. Do not calculate it. "
        "Return exactly <calc>EXPRESSION</calc>. Include every quantity and relationship needed.\n\n"
        + question
    )


def solve_planned_expression(raw: str) -> str | None:
    match = re.search(r"<calc>\s*([^<>]+?)\s*</calc>", raw, re.I | re.S)
    if not match:
        return None
    value = evaluate_arithmetic(match.group(1).strip())
    if value is None:
        return None
    if value.denominator == 1:
        return str(value.numerator)
    return str(round(float(value), 10)).rstrip("0").rstrip(".")


def solve_model_planned_arithmetic(model, tokenizer, question: str) -> str | None:
    """Let the local model plan, then deterministically verify and execute its arithmetic."""
    from mlx_lm import generate

    instruction = arithmetic_planner_prompt(question)
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "You produce safe arithmetic expressions for a verified calculator."},
        {"role": "user", "content": instruction},
    ], tokenize=False, add_generation_prompt=True)
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=192, verbose=False).strip()
    return solve_planned_expression(raw)


def solve_exact(question: str) -> int | None:
    arithmetic = re.fullmatch(
        r"Compute \((\d+) \* (\d+)\) - \((\d+) \* (\d+)\)\. Give only the integer\.",
        question,
    )
    if arithmetic:
        a, b, c, d = map(int, arithmetic.groups())
        return a * b - c * d

    recurrence = re.fullmatch(
        r"x0=(\d+); x\(n\+1\)=\((\d+)\*x\(n\)\+(\d+)\) mod (\d+)\. Find x(\d+)\. Give only the integer\.",
        question,
    )
    if recurrence:
        value, multiplier, addition, modulus, steps = map(int, recurrence.groups())
        for _ in range(steps):
            value = (multiplier * value + addition) % modulus
        return value

    grid = re.fullmatch(
        r"Count Right/Up paths from \(0,0\) to \((\d+),(\d+)\) avoiding (.+)\. Give only the integer\.",
        question,
    )
    if grid:
        width, height = map(int, grid.groups()[:2])
        blocked = set(map(tuple, ast.literal_eval(grid.group(3))))
        ways = [[0] * (height + 1) for _ in range(width + 1)]
        ways[0][0] = 1
        for x in range(width + 1):
            for y in range(height + 1):
                if (x, y) == (0, 0):
                    continue
                ways[x][y] = 0 if (x, y) in blocked else (
                    (ways[x - 1][y] if x else 0) + (ways[x][y - 1] if y else 0)
                )
        return ways[width][height]

    subset = re.fullmatch(
        r"Positions are distinct\. How many subsets of (\[.*\]) sum to (\d+)\? Give only the integer\.",
        question,
    )
    if subset:
        values = ast.literal_eval(subset.group(1))
        target = int(subset.group(2))
        counts = [0] * (target + 1)
        counts[0] = 1
        for value in values:
            for total in range(target, value - 1, -1):
                counts[total] += counts[total - value]
        return counts[target]
    return None
