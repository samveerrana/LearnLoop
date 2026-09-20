from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who", "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    }


@dataclass(frozen=True)
class Memory:
    id: int
    question: str
    correction: str
    evidence: str
    score: float = 0.0


class MemoryStore:
    """Inspectable, reversible memory. The base model is never overwritten."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                bad_answer TEXT NOT NULL,
                correction TEXT NOT NULL,
                evidence TEXT NOT NULL DEFAULT '',
                approved INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_url TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(topic, source_url)
            )
            """
        )
        self.db.commit()

    def add(self, question: str, bad_answer: str, correction: str, evidence: str = "") -> int:
        created = datetime.now(timezone.utc).isoformat()
        cursor = self.db.execute(
            "INSERT INTO corrections(question,bad_answer,correction,evidence,created_at) VALUES(?,?,?,?,?)",
            (question.strip(), bad_answer.strip(), correction.strip(), evidence.strip(), created),
        )
        self.db.commit()
        return int(cursor.lastrowid)

    def search(self, query: str, limit: int = 4) -> list[Memory]:
        query_tokens = _tokens(query)
        rows = self.db.execute(
            "SELECT id,question,correction,evidence FROM corrections WHERE approved=1"
        ).fetchall()
        ranked: list[Memory] = []
        for row in rows:
            candidate_tokens = _tokens(row[1] + " " + row[2])
            union = query_tokens | candidate_tokens
            score = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
            if score > 0:
                ranked.append(Memory(*row, score=score))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM corrections").fetchone()[0])

    def add_knowledge(self, topic: str, title: str, content: str, source_url: str) -> None:
        created = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """INSERT OR REPLACE INTO knowledge(topic,title,content,source_url,created_at)
               VALUES(?,?,?,?,?)""",
            (topic.strip(), title.strip(), content.strip(), source_url.strip(), created),
        )
        self.db.commit()

    def search_knowledge(self, query: str, limit: int = 3) -> list[dict[str, str]]:
        query_tokens = _tokens(query)
        rows = self.db.execute(
            "SELECT topic,title,content,source_url FROM knowledge"
        ).fetchall()
        ranked: list[tuple[float, dict[str, str]]] = []
        for topic, title, content, source_url in rows:
            heading_tokens = _tokens(topic + " " + title)
            overlap = len(query_tokens & heading_tokens)
            if overlap:
                score = overlap / max(1, len(query_tokens))
                ranked.append(
                    (score, {"topic": topic, "title": title, "content": content, "source_url": source_url})
                )
        return [item for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True)[:limit]]

    def knowledge_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0])

    def export_jsonl(self, destination: Path) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.db.execute(
            "SELECT question,correction FROM corrections WHERE approved=1 ORDER BY id"
        ).fetchall()
        with destination.open("w", encoding="utf-8") as handle:
            for question, correction in rows:
                sample = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": correction},
                    ]
                }
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return len(rows)
