import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from torch.utils.data import Dataset

from falldet.data.video_dataset import label2idx
from falldet.schemas import PreferenceConfig, from_dictconfig_dpo
from falldet.training.preference_setup import build_negative_selectors
from falldet.training.preferences import RandomNegativeSelector
from falldet.training.scored import (
    SOURCE_FALLBACK,
    SOURCE_SCORE,
    ScoredNegativeSelector,
    load_label_scores,
)


def _scores(**overrides):
    scores = {label: -10.0 for label in label2idx}
    scores.update(overrides)
    return scores


def test_rejects_highest_scoring_wrong_label():
    selector = ScoredNegativeSelector(
        ["fall", "walk"],
        [_scores(fall=-1.0, jump=-2.0), _scores(fall=-0.5, walk=-3.0, standing=-1.0)],
        RandomNegativeSelector(tuple(label2idx), seed=0),
    )

    assert selector.negative_labels == ("jump", "fall")
    assert selector.select(0, "fall") == "jump"
    assert selector.sources == (SOURCE_SCORE, SOURCE_SCORE)
    assert selector.margins == (1.0, -2.5)
    assert selector.correct == (True, False)
    assert selector.summary("train") == {
        "train_score_coverage": 1.0,
        "train_scoring_model_accuracy": 0.5,
        "train_score_margin_mean": -0.75,
    }


def test_rows_without_scores_use_fallback():
    fallback = RandomNegativeSelector(tuple(label2idx), seed=0)
    selector = ScoredNegativeSelector(["fall", "walk"], [None, _scores(fall=-1.0)], fallback)

    assert selector.negative_labels[0] == fallback.select(0, "fall")
    assert selector.sources == (SOURCE_FALLBACK, SOURCE_SCORE)
    assert selector.summary("validation")["validation_score_coverage"] == 0.5


def test_load_label_scores_requires_every_label(tmp_path):
    path = tmp_path / "scores.jsonl"
    rows = [
        {"type": "metadata", "config": {}},
        {"type": "prediction", "idx": 0, "label_logprobs": {"fall": -1.0}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="every label"):
        load_label_scores(path)


class SegmentDataset(Dataset):
    dataset_name = "OOPS"

    def __init__(self, rows):
        self.video_segments = [
            {"video_path": video, "start": 0.0, "end": 1.0, "label_str": label}
            for video, label in rows
        ]

    def __len__(self):
        return len(self.video_segments)


def _write_scores(path, rows, prompt, adapter):
    lines = [{"type": "metadata", "config": {"prompt": prompt, "lora": {"path": adapter}}}]
    for idx, (video, label, scores) in enumerate(rows):
        lines.append(
            {
                "type": "prediction",
                "idx": idx,
                "dataset": "OOPS",
                "video_path": video,
                "start_time": 0.0,
                "end_time": 1.0,
                "label_str": label,
                "label_logprobs": scores,
            }
        )
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_build_scores_preferences_from_config(tmp_path, caplog):
    path = tmp_path / "scores.jsonl"
    config_dir = str(Path(__file__).parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="dpo_config",
            overrides=["preference=scores", f"preference.train_scores_path={path}"],
        )
    config = from_dictconfig_dpo(cfg)
    _write_scores(
        path,
        [("a", "fall", _scores(fall=-1.0, lying=-0.5)), ("b", "walk", _scores(walk=-1.0))],
        prompt=json.loads(config.prompt.model_dump_json()),
        adapter="/elsewhere/adapter",
    )
    train = SegmentDataset([("a", "fall"), ("b", "walk"), ("c", "jump")])
    validation = SegmentDataset([("d", "fall")])

    setup = build_negative_selectors(config, train, validation)

    assert setup.train.select(0, "fall") == "lying"
    assert setup.train.sources == (SOURCE_SCORE, SOURCE_SCORE, SOURCE_FALLBACK)
    assert setup.validation.sources == (SOURCE_FALLBACK,)
    assert setup.summary["train_score_coverage"] == pytest.approx(2 / 3)
    assert setup.summary["train_scoring_model_accuracy"] == 0.5
    assert [row["split"] for row in setup.manifest] == ["train"] * 3 + ["validation"]
    assert "come from adapter /elsewhere/adapter" in caplog.text
    assert "different prompt config" not in caplog.text


def test_scores_preference_requires_train_scores_path():
    with pytest.raises(ValueError, match="require train_scores_path"):
        PreferenceConfig(strategy="scores")
