from __future__ import annotations

from pathlib import Path

from mlx_lm import generate, load

from .memory import MemoryStore
from .prompting import SYSTEM_PROMPT, inference_user_content
from .research import WikipediaResearcher, should_research


class LocalModel:
    def __init__(self, model_path: Path, memory: MemoryStore, adapter_path: Path | None = None):
        self.model_path = model_path
        self.memory = memory
        self.model, self.tokenizer = load(
            str(model_path), adapter_path=str(adapter_path) if adapter_path else None
        )
        self.researcher = WikipediaResearcher(memory)

    def answer(
        self,
        question: str,
        max_tokens: int = 350,
        auto_research: bool = False,
        thinking: bool = False,
    ) -> str:
        memories = self.memory.search(question)
        knowledge = self.memory.search_knowledge(question)
        if auto_research and not knowledge and should_research(question):
            try:
                self.researcher.research(question)
                knowledge = self.memory.search_knowledge(question)
            except Exception as error:
                print(f"[research unavailable: {error}]")
        memory_text = "\n".join(
            f"- Earlier correction: {item.correction}"
            + (f" (evidence: {item.evidence})" if item.evidence else "")
            for item in memories
        ) or "- No relevant corrections yet."
        knowledge_text = "\n\n".join(
            f"SOURCE: {item['title']} — {item['source_url']}\n{item['content'][:2500]}"
            for item in knowledge
        ) or "No external research supplied; use internal learned knowledge."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": inference_user_content(question, memory_text, knowledge_text),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        ).strip()
