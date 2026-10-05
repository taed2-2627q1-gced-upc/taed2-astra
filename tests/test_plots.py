"""Sustainability plots from CodeCarbon emissions data."""

# redefined-outer-name: pylint false positive on pytest fixtures injected as arguments.
# protected-access: _benchmark_runs is a private helper tested on purpose.
# pylint: disable=redefined-outer-name,protected-access

import pandas as pd
import pytest

from taed2_astra.visualization import plots


@pytest.fixture
def emissions() -> pd.DataFrame:
    """Return benchmark rows mixed with unrelated stages and an older run."""
    return pd.DataFrame(
        {
            "project_name": [
                "logistic_regression",
                "train-hist_gradient_boosting",
                "hist_gradient_boosting",
                "ensemble_soft",
                "ensemble_stacking",
                "logistic_regression",
                "inference-hist_gradient_boosting",
            ],
            "duration": [10, 11, 20, 30, 40, 15, 5],
            "energy_consumed": [0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.01],
        }
    )


def test_benchmark_runs_selects_latest_candidate_run_in_display_order(emissions):
    """Repeated runs use the newest candidate row and exclude other pipeline stages."""
    runs = plots._benchmark_runs(emissions)

    assert runs.index.tolist() == list(plots.BENCHMARK_MODELS)
    assert runs.loc["logistic_regression", "energy_consumed"] == 0.15
    assert runs.loc["hist_gradient_boosting", "duration"] == 20


def test_plot_functions_write_both_pngs(emissions, tmp_path, monkeypatch):
    """Each requested chart is saved in the configured figures directory."""
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)

    energy_path = plots.plot_energy_consumed(emissions)
    duration_path = plots.plot_duration_vs_energy(emissions)

    assert energy_path.name == "energy_consumed_per_model.png"
    assert duration_path.name == "duration_vs_energy.png"
    assert energy_path.stat().st_size > 0
    assert duration_path.stat().st_size > 0


def test_benchmark_runs_rejects_missing_candidate(emissions):
    """A chart must not silently omit one of the four intended model comparisons."""
    emissions = emissions[emissions["project_name"] != "ensemble_soft"]

    with pytest.raises(ValueError, match="ensemble_soft"):
        plots._benchmark_runs(emissions)
