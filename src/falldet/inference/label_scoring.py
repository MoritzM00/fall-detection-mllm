"""Teacher-forced log-probabilities of every class label from vLLM prompt logprobs.

Each clip becomes one request per label whose prompt ends with that label's DPO
completion. Only the label-dependent tail is scored: completion tokens before the
first position where the candidates differ are identical for every label and
cancel in any comparison between labels.

Prompt logprobs are read from the end of the list, so a list that is shorter than
the prompt (tokens served from a prefix cache) still aligns; a scored token without
a logprob yields no score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class _Logprob(Protocol):
    logprob: float


class ScoredOutput(Protocol):
    """The fields of a vLLM ``RequestOutput`` used for scoring."""

    prompt_token_ids: Sequence[int] | None
    prompt_logprobs: Sequence[Mapping[int, _Logprob] | None] | None


@dataclass(frozen=True)
class LabelCompletions:
    """Tokenized DPO completions for all labels plus their shared prefix length."""

    labels: tuple[str, ...]
    texts: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    shared_prefix: int

    @classmethod
    def build(
        cls, labels: Sequence[str], texts: Sequence[str], token_ids: Sequence[Sequence[int]]
    ) -> LabelCompletions:
        if not (len(labels) == len(texts) == len(token_ids)) or len(labels) < 2:
            raise ValueError("Need matching texts and token ids for at least two labels")
        ids = tuple(tuple(int(token) for token in row) for row in token_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("Label completions must tokenize to distinct sequences")
        shared = 0
        while all(len(row) > shared for row in ids) and len({row[shared] for row in ids}) == 1:
            shared += 1
        return cls(tuple(labels), tuple(texts), ids, shared)


def tail_logprob(
    output: ScoredOutput, completion_ids: Sequence[int], shared_prefix: int
) -> float | None:
    """Sum the prompt logprobs of the label-dependent completion tokens.

    Returns None if vLLM did not compute a logprob for every scored token (served
    from the prefix cache). Raises if the prompt does not end with the completion,
    i.e. the text prompt tokenized differently than the DPO completion.
    """

    prompt_ids = list(output.prompt_token_ids or [])
    completion = list(completion_ids)
    if prompt_ids[-len(completion) :] != completion:
        raise ValueError(
            "Prompt does not end with the label completion tokens; the prompt/completion "
            "boundary tokenized differently than in DPO"
        )
    logprobs = list(output.prompt_logprobs or [])
    scored = completion[shared_prefix:]
    if len(logprobs) < len(scored):
        return None
    total = 0.0
    for token, entry in zip(scored, logprobs[len(logprobs) - len(scored) :], strict=True):
        if entry is None or token not in entry:
            return None
        total += float(entry[token].logprob)
    return total


def score_clip(
    outputs: Sequence[ScoredOutput], completions: LabelCompletions
) -> dict[str, float] | None:
    """Label -> log-probability for one clip, or None if any label lacks a score."""

    if len(outputs) != len(completions.labels):
        raise ValueError("Need one output per label")
    scores: dict[str, float] = {}
    for label, ids, output in zip(completions.labels, completions.token_ids, outputs, strict=True):
        value = tail_logprob(output, ids, completions.shared_prefix)
        if value is None:
            return None
        scores[label] = value
    return scores


def label_probabilities(scores: Mapping[str, float]) -> dict[str, float]:
    """Normalize label log-probabilities into a distribution over the labels."""

    top = max(scores.values())
    weights = {label: math.exp(value - top) for label, value in scores.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}
