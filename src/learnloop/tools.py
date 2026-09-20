from __future__ import annotations

import ast
import json
import subprocess
import sys


ALLOWED_CALLS = {"range", "len", "sum", "min", "max", "abs", "enumerate", "sorted", "set", "list", "dict", "int"}
ALLOWED_METHODS = {"append", "add", "index", "count", "get"}
BLOCKED_NODES = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)


class UnsafeCode(ValueError):
    pass


def validate_calculation(code: str) -> None:
    tree = ast.parse(code)
    local_functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODES):
            raise UnsafeCode(f"blocked syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCode("private names are blocked")
        if isinstance(node, ast.Attribute) and node.attr not in ALLOWED_METHODS:
            raise UnsafeCode(f"blocked attribute: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in ALLOWED_CALLS | local_functions:
                raise UnsafeCode(f"blocked call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr not in ALLOWED_METHODS:
                raise UnsafeCode(f"blocked method: {node.func.attr}")


def execute_calculation(code: str, timeout: float = 5.0) -> int | float | str:
    validate_calculation(code)
    wrapper = """
import json, sys
code = sys.stdin.read()
safe = {name: getattr(__builtins__, name) for name in
        ['range','len','sum','min','max','abs','enumerate','sorted','set','list','dict','int']}
scope = {'__builtins__': safe}
exec(compile(code, '<calculation>', 'exec'), scope, scope)
if 'answer' not in scope:
    raise ValueError('calculation did not assign answer')
print(json.dumps(scope['answer']))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", wrapper],
        input=code,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return json.loads(completed.stdout)

