from pathlib import Path

from learnloop.release_check import check


def test_release_check_accepts_minimal_safe_tree(tmp_path: Path) -> None:
    for name in ("README.md", "LICENSE", ".gitignore", "PUBLIC_V1_CHECKLIST.md", "pyproject.toml"):
        tmp_path.joinpath(name).write_text("safe")
    assert check(tmp_path) == {"passed": True, "errors": []}


def test_release_check_rejects_home_path_and_private_root_file(tmp_path: Path) -> None:
    for name in ("LICENSE", ".gitignore", "PUBLIC_V1_CHECKLIST.md", "pyproject.toml"):
        tmp_path.joinpath(name).write_text("safe")
    tmp_path.joinpath("README.md").write_text("/Users/example/private")
    tmp_path.joinpath("secret.key").write_text("not-a-real-key")
    result = check(tmp_path)
    assert not result["passed"]
    assert len(result["errors"]) == 2
