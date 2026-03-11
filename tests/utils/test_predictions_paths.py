from pathlib import Path

from falldet.utils.predictions import prediction_jsonl_path, prediction_jsonl_relpath
from falldet.utils.wandb import get_prediction_output_path


def test_prediction_jsonl_relpath_is_project_scoped():
    assert prediction_jsonl_relpath("demo-project", "abc123") == Path(
        "predictions/demo-project/abc123.jsonl"
    )


def test_prediction_jsonl_path_joins_output_root():
    output_root = Path("outputs")
    assert prediction_jsonl_path(output_root, "demo-project", "abc123") == Path(
        "outputs/predictions/demo-project/abc123.jsonl"
    )


def test_get_prediction_output_path_uses_canonical_location():
    assert get_prediction_output_path("outputs", "demo-project", "abc123") == Path(
        "outputs/predictions/demo-project/abc123.jsonl"
    )
