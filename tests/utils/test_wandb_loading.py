from pathlib import Path

import pytest

from falldet.utils.predictions import save_predictions_jsonl
from falldet.utils.wandb import load_run_from_wandb


def test_load_run_from_wandb_prefers_canonical_local_file(tmp_path, monkeypatch):
    project = "demo-project"
    run_id = "abc123"
    output_root = tmp_path / "outputs"
    local_path = output_root / "predictions" / project / f"{run_id}.jsonl"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    config = {"model": {"name": "demo"}}
    predictions = [{"label_str": "fall", "predicted_label": "fall", "dataset": "wanfall"}]
    save_predictions_jsonl(
        output_path=local_path,
        config=config,
        predictions=predictions,
        wandb_run_id=run_id,
    )

    def unexpected_api():
        raise AssertionError("wandb.Api should not be called when a local file exists")

    monkeypatch.setattr("falldet.utils.wandb.wandb.Api", unexpected_api)

    loaded_config, loaded_predictions = load_run_from_wandb(
        run_id,
        project=project,
        output_root=output_root,
    )

    assert loaded_config == config
    assert loaded_predictions == [
        {
            "type": "prediction",
            "idx": 0,
            "label_str": "fall",
            "predicted_label": "fall",
            "dataset": "wanfall",
        }
    ]


def test_load_run_from_wandb_validates_local_run_id(tmp_path):
    project = "demo-project"
    output_root = tmp_path / "outputs"
    local_path = output_root / "predictions" / project / "abc123.jsonl"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    save_predictions_jsonl(
        output_path=local_path,
        config={},
        predictions=[{"label_str": "fall", "predicted_label": "fall"}],
        wandb_run_id="different-id",
    )

    with pytest.raises(ValueError, match="different-id"):
        load_run_from_wandb("abc123", project=project, output_root=output_root)


def test_load_run_from_wandb_requires_entity_only_when_download_needed(tmp_path, monkeypatch):
    project = "demo-project"
    run_id = "abc123"
    output_root = tmp_path / "outputs"
    local_path = output_root / "predictions" / project / f"{run_id}.jsonl"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    save_predictions_jsonl(
        output_path=local_path,
        config={},
        predictions=[{"label_str": "fall", "predicted_label": "fall"}],
        wandb_run_id=run_id,
    )

    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    config, predictions = load_run_from_wandb(run_id, project=project, output_root=output_root)
    assert config == {}
    assert predictions[0]["predicted_label"] == "fall"


def test_load_run_from_wandb_requires_entity_for_download(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    with pytest.raises(ValueError, match="WANDB_ENTITY"):
        load_run_from_wandb("abc123", project="demo-project", output_root=tmp_path / "outputs")


class _FakeFile:
    def __init__(self, name: str, source: Path):
        self.name = name
        self._source = source

    def download(self, root: str, replace: bool = True) -> None:
        destination = Path(root) / self.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._source.read_bytes())


class _FakeRun:
    def __init__(self, files: list[_FakeFile]):
        self._files = files

    def files(self) -> list[_FakeFile]:
        return self._files


class _FakeApi:
    def __init__(self, run: _FakeRun):
        self._run = run
        self.paths: list[str] = []

    def run(self, path: str) -> _FakeRun:
        self.paths.append(path)
        return self._run


def test_load_run_from_wandb_downloads_to_canonical_path(tmp_path, monkeypatch):
    project = "demo-project"
    entity = "demo-entity"
    run_id = "abc123"
    output_root = tmp_path / "outputs"
    download_source = tmp_path / "source.jsonl"

    save_predictions_jsonl(
        output_path=download_source,
        config={"dataset": {"name": "wanfall"}},
        predictions=[{"label_str": "fall", "predicted_label": "fall", "dataset": "wanfall"}],
        wandb_run_id=run_id,
    )

    fake_run = _FakeRun(
        [
            _FakeFile(
                f"predictions/{project}/{run_id}.jsonl",
                download_source,
            )
        ]
    )
    fake_api = _FakeApi(fake_run)
    monkeypatch.setattr("falldet.utils.wandb.wandb.Api", lambda: fake_api)

    config, predictions = load_run_from_wandb(
        run_id,
        project=project,
        entity=entity,
        output_root=output_root,
    )

    canonical_path = output_root / "predictions" / project / f"{run_id}.jsonl"
    assert canonical_path.exists()
    assert fake_api.paths == [f"{entity}/{project}/{run_id}"]
    assert config == {"dataset": {"name": "wanfall"}}
    assert predictions[0]["predicted_label"] == "fall"


def test_load_run_from_wandb_downloads_legacy_filename_when_metadata_matches(tmp_path, monkeypatch):
    project = "demo-project"
    entity = "demo-entity"
    run_id = "abc123"
    output_root = tmp_path / "outputs"
    legacy_source = tmp_path / "legacy.jsonl"

    save_predictions_jsonl(
        output_path=legacy_source,
        config={"legacy": True},
        predictions=[{"label_str": "fall", "predicted_label": "fall"}],
        wandb_run_id=run_id,
    )

    fake_run = _FakeRun([_FakeFile("legacy-name.jsonl", legacy_source)])
    monkeypatch.setattr("falldet.utils.wandb.wandb.Api", lambda: _FakeApi(fake_run))

    config, predictions = load_run_from_wandb(
        run_id,
        project=project,
        entity=entity,
        output_root=output_root,
    )

    assert config == {"legacy": True}
    assert predictions[0]["predicted_label"] == "fall"


def test_load_run_from_wandb_ignores_wrong_metadata_before_matching_file(tmp_path, monkeypatch):
    project = "demo-project"
    entity = "demo-entity"
    run_id = "abc123"
    output_root = tmp_path / "outputs"
    wrong_source = tmp_path / "wrong.jsonl"
    correct_source = tmp_path / "correct.jsonl"

    save_predictions_jsonl(
        output_path=wrong_source,
        config={"wrong": True},
        predictions=[{"label_str": "fall", "predicted_label": "other"}],
        wandb_run_id="wrong-id",
    )
    save_predictions_jsonl(
        output_path=correct_source,
        config={"correct": True},
        predictions=[{"label_str": "fall", "predicted_label": "fall"}],
        wandb_run_id=run_id,
    )

    fake_run = _FakeRun(
        [
            _FakeFile("other.jsonl", wrong_source),
            _FakeFile(f"predictions/{project}/{run_id}.jsonl", correct_source),
        ]
    )
    monkeypatch.setattr("falldet.utils.wandb.wandb.Api", lambda: _FakeApi(fake_run))

    config, predictions = load_run_from_wandb(
        run_id,
        project=project,
        entity=entity,
        output_root=output_root,
    )

    assert config == {"correct": True}
    assert predictions[0]["predicted_label"] == "fall"
