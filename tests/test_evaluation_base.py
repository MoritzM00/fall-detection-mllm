import json
from types import SimpleNamespace

from falldet.evaluation.base import save_evaluation_results


def test_save_evaluation_results_scopes_run_outputs_under_project(tmp_path):
    output_dir = tmp_path / "outputs"
    run = SimpleNamespace(name="demo-run", project="demo-project")

    save_evaluation_results({"accuracy": 1.0}, None, str(output_dir), run)

    results_file = output_dir / "evaluation_results" / "demo-project" / "test_results_demo-run.json"
    assert results_file.exists()
    assert json.loads(results_file.read_text()) == {"accuracy": 1.0}


def test_save_evaluation_results_without_run_keeps_base_output_dir(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr("falldet.evaluation.base.time.strftime", lambda _fmt: "20260314-123456")

    save_evaluation_results({"accuracy": 1.0}, None, str(output_dir), None)

    results_file = output_dir / "evaluation_results" / "test_results_results_20260314-123456.json"
    assert results_file.exists()
    assert json.loads(results_file.read_text()) == {"accuracy": 1.0}
