from types import SimpleNamespace

import pytest

from falldet.inference.label_scoring import (
    LabelCompletions,
    label_probabilities,
    score_clip,
    tail_logprob,
)

PROMPT = [101, 102, 103]
# "The best answer is: <label><eos>\n" with a shared 2-token answer prefix
COMPLETIONS = LabelCompletions.build(
    ["fall", "fallen", "walk"],
    ["a", "b", "c"],
    [[7, 8, 20, 99, 10], [7, 8, 20, 21, 99, 10], [7, 8, 30, 99, 10]],
)


def _output(completion, logprobs, cached=0):
    """Fake vLLM output: one logprob per computed prompt token, first entry None."""
    ids = PROMPT + list(completion)
    entries = [None] + [
        {token: SimpleNamespace(logprob=lp)} for token, lp in zip(ids[1:], logprobs)
    ]
    return SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=[None] + entries[1 + cached :])


def test_shared_prefix_counts_tokens_common_to_all_labels():
    assert COMPLETIONS.shared_prefix == 2


def test_build_rejects_identical_tokenizations():
    with pytest.raises(ValueError, match="distinct"):
        LabelCompletions.build(["fall", "walk"], ["a", "b"], [[1, 2], [1, 2]])


def test_tail_logprob_sums_only_label_dependent_tokens():
    ids = COMPLETIONS.token_ids[0]
    logprobs = [-9.0] * (len(PROMPT) - 1) + [-5.0, -5.0, -1.0, -0.5, -0.25]
    assert tail_logprob(_output(ids, logprobs), ids, COMPLETIONS.shared_prefix) == -1.75


def test_tail_logprob_ignores_cached_prompt_tokens():
    ids = COMPLETIONS.token_ids[0]
    logprobs = [-9.0] * (len(PROMPT) - 1) + [-5.0, -5.0, -1.0, -0.5, -0.25]
    output = _output(ids, logprobs, cached=len(PROMPT) + 1)
    assert tail_logprob(output, ids, COMPLETIONS.shared_prefix) == -1.75


def test_tail_logprob_is_none_when_scored_tokens_were_cached():
    ids = COMPLETIONS.token_ids[0]
    logprobs = [-1.0] * (len(PROMPT) + len(ids) - 1)
    # Only the last two tokens were computed; three are scored
    output = _output(ids, logprobs, cached=len(PROMPT) + len(ids) - 3)
    assert tail_logprob(output, ids, COMPLETIONS.shared_prefix) is None


def test_tail_logprob_rejects_prompt_not_ending_with_completion():
    ids = COMPLETIONS.token_ids[0]
    output = _output(COMPLETIONS.token_ids[2], [-1.0] * 7)
    with pytest.raises(ValueError, match="tokenized differently"):
        tail_logprob(output, ids, COMPLETIONS.shared_prefix)


def test_score_clip_returns_none_if_any_label_is_missing():
    outputs = [_output(ids, [-1.0] * (len(PROMPT) + len(ids) - 1)) for ids in COMPLETIONS.token_ids]
    scores = score_clip(outputs, COMPLETIONS)
    assert scores == {"fall": -3.0, "fallen": -4.0, "walk": -3.0}

    ids = COMPLETIONS.token_ids[1]
    outputs[1] = _output(ids, [-1.0] * (len(PROMPT) + len(ids) - 1), cached=len(PROMPT) + 4)
    assert score_clip(outputs, COMPLETIONS) is None


def test_label_probabilities_normalize():
    probabilities = label_probabilities({"fall": 0.0, "walk": 0.0})
    assert probabilities == {"fall": 0.5, "walk": 0.5}
