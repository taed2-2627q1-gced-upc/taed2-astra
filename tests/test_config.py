"""The layout and params contract the rest of the code depends on."""

from taed2_astra.config import PROJECT_ROOT, load_params


def test_project_root_is_the_repository():
    """PROJECT_ROOT must resolve to the folder holding pyproject.toml."""
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_params_expose_every_section_the_pipeline_reads():
    """Each pipeline stage reads one of these sections by name."""
    params = load_params()
    for section in ("dataset", "prepare", "train", "benchmark", "evaluate", "mlflow"):
        assert section in params, f"params.yaml is missing the '{section}' section"
