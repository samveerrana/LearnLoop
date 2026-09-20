from pathlib import Path

from learnloop.memory import MemoryStore
from learnloop.tools import UnsafeCode, execute_calculation
from learnloop.adapter_cache import AdapterCache
from learnloop.quiz import generate
from learnloop.training_data import build_training_splits
import pytest


def test_correction_is_retrievable_and_exportable(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add(
        "What does glorp do in my game?",
        "It causes damage.",
        "Glorp restores 10 health.",
        "game design document",
    )

    results = store.search("How much health does glorp restore?")
    assert results[0].correction == "Glorp restores 10 health."

    output = tmp_path / "train.jsonl"
    assert store.export_jsonl(output) == 1
    assert '"role": "assistant"' in output.read_text()


def test_researched_knowledge_survives_a_restart(tmp_path: Path):
    database = tmp_path / "memory.sqlite3"
    first = MemoryStore(database)
    first.add_knowledge(
        "history of the US",
        "History of the United States",
        "A verified history summary.",
        "https://example.test/history",
    )
    first.db.close()

    restarted = MemoryStore(database)
    result = restarted.search_knowledge("Tell me the history of the US")
    assert result[0]["content"] == "A verified history summary."
    assert restarted.knowledge_count() == 1


def test_unrelated_knowledge_is_not_attached(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add_knowledge("US history", "History of the United States", "History text", "https://example.test")
    assert store.search_knowledge("How many paths are in this robot grid?") == []


def test_safe_calculation_tool():
    assert execute_calculation("answer = sum(range(11))") == 55


def test_calculation_tool_blocks_file_access():
    with pytest.raises(UnsafeCode):
        execute_calculation("answer = open('/tmp/nope').read()")


def test_quiz_has_1000_separate_answers(tmp_path: Path):
    generate(tmp_path)
    assert len((tmp_path / "questions.jsonl").read_text().splitlines()) == 1000
    assert len((tmp_path / "private-answers.jsonl").read_text().splitlines()) == 1000
    assert "answer" not in (tmp_path / "questions.jsonl").read_text().splitlines()[0]


def test_adapter_cache_limit(tmp_path: Path):
    cache = AdapterCache(tmp_path, limit=5)
    first = tmp_path / "old"
    first.mkdir()
    (first / "weights").write_bytes(b"123456")
    assert cache.enforce_limit() == [first]


def test_training_and_locked_test_are_separate(tmp_path: Path):
    quiz = tmp_path / "quiz"
    generate(quiz)
    output = tmp_path / "training"
    assert build_training_splits(quiz, output) == {"train": 700, "valid": 100, "test": 100}
    assert output.joinpath("train.jsonl").read_text() != output.joinpath("test.jsonl").read_text()
