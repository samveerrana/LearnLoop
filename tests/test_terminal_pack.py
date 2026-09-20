import pytest

from learnloop.terminal_pack import choose_model


def test_requested_safe_installed_model() -> None:
    models = [{"name": "qwen3:4b", "size": 3_000_000_000}]
    assert choose_model(models, "qwen3:4b", memory_bytes=16 * 1024**3) == "qwen3:4b"


def test_refuses_large_local_model() -> None:
    models = [{"name": "gpt-oss:20b", "size": 13_000_000_000}]
    with pytest.raises(SystemExit, match="safety limit"):
        choose_model(models, "gpt-oss:20b", memory_bytes=16 * 1024**3)


def test_larger_machine_can_select_larger_model() -> None:
    models = [{"name": "gpt-oss:20b", "size": 13_000_000_000}]
    assert choose_model(models, "gpt-oss:20b", memory_bytes=32 * 1024**3) == "gpt-oss:20b"
