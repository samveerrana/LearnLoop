from __future__ import annotations

import re
import ast

from .model import LocalModel
from .tools import execute_calculation


CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def tool_answer(model: LocalModel, question: str) -> tuple[str, str]:
    request = f"""Solve the problem below by writing a small Python calculation.
Return only one Python code block. Do not import anything. The code must assign the final integer to a variable named answer.
The checker rejects hard-coded answers such as `answer = 16`; calculate it algorithmically with loops or comprehensions.
You may use loops, functions, lists, dictionaries, sets, range, len, sum, min, max, abs, enumerate, sorted, and collection methods.

PROBLEM:
{question}"""
    proposal = model.answer(request, max_tokens=900, auto_research=False, thinking=False)
    match = CODE_RE.search(proposal)
    code = match.group(1).strip() if match else proposal.strip()
    if code.lower().startswith("python\n"):
        code = code.split("\n", 1)[1]
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "answer" for target in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            raise ValueError("model hard-coded the answer instead of calculating it")
    result = execute_calculation(code)
    return f"FINAL: {result}", code
