"""Tokenizer-aware chunking for long documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ChunkingError(ValueError):
    """Raised when text cannot be chunked safely."""


class TokenizerLike(Protocol):
    """Small tokenizer surface required by the chunker."""

    name_or_path: str

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def num_special_tokens_to_add(self, *, pair: bool = False) -> int: ...

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]: ...


@dataclass(frozen=True)
class TokenChunk:
    """A model-ready token window and its non-special-token length."""

    input_ids: tuple[int, ...]
    content_token_count: int


def chunk_text(
    tokenizer: TokenizerLike,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> list[TokenChunk]:
    """Tokenize once and create overlapping model-ready windows.

    ``stride`` is the number of content tokens repeated between adjacent windows.
    Special-token capacity is reserved before the windows are created.
    """

    if not isinstance(text, str) or not text.strip():
        raise ChunkingError("Text must be a non-empty string.")
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_capacity = max_length - special_tokens
    if content_capacity < 1:
        raise ChunkingError("max_length leaves no room for content tokens.")
    if stride < 0 or stride >= content_capacity:
        raise ChunkingError("stride must be non-negative and smaller than content capacity.")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ChunkingError("Tokenizer produced no content tokens.")

    chunks: list[TokenChunk] = []
    step = content_capacity - stride
    for start in range(0, len(token_ids), step):
        content_ids = token_ids[start : start + content_capacity]
        model_ids = tokenizer.build_inputs_with_special_tokens(content_ids)
        if len(model_ids) > max_length:
            raise ChunkingError("Tokenizer added more special tokens than it reported.")
        chunks.append(TokenChunk(tuple(model_ids), len(content_ids)))
        if start + content_capacity >= len(token_ids):
            break
    return chunks


def split_paragraphs(text: str) -> list[str]:
    """Return non-empty paragraphs while retaining sentence order."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    return blocks or ([text.strip()] if text.strip() else [])
