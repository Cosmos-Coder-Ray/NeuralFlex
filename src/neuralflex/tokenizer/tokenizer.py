from __future__ import annotations


class NeuralFlexTokenizer:
    """In-repo tokenizer stub with no runtime network dependency."""

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("Tokenizer runtime will be implemented in Milestone 3.")

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError("Tokenizer runtime will be implemented in Milestone 3.")
