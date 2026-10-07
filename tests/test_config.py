"""The layout and params contract the rest of the code depends on."""

from taed2_astra.config import PROJECT_ROOT, load_params, write_json


def test_project_root_is_the_repository():
    """PROJECT_ROOT must resolve to the folder holding pyproject.toml."""
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_params_expose_every_section_the_pipeline_reads():
    """Each pipeline stage reads one of these sections by name."""
    params = load_params()
    sections = ("dataset", "prepare", "train", "benchmark", "evaluate", "mlflow", "validation", "model_quality")
    for section in (*sections, "fairness"):
        assert section in params, f"params.yaml is missing the '{section}' section"


def test_json_metrics_are_written_with_lf_endings(tmp_path):
    """dvc.lock hashes the bytes on disk; CRLF on Windows would not match the LF copy every clone checks out."""
    path = tmp_path / "metrics.json"
    write_json(path, {"recall": 0.5, "nested": {"n": 1}})
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"}\n")
