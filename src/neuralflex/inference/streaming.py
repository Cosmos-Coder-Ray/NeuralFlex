from __future__ import annotations

from collections.abc import Iterator


class StreamingGenerator:
    """Streaming token generator stub for local inference."""

    def stream(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError("Streaming decode loop is planned for Milestone 4.")
