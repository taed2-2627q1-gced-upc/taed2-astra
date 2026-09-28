"""Sustainability figures for the report and model card."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from taed2_astra.config import EMISSIONS_PATH, FIGURES_DIR

BENCHMARK_MODELS = ("logistic_regression", "hist_gradient_boosting", "ensemble_soft", "ensemble_stacking")
REQUIRED_COLUMNS = ("project_name", "duration", "energy_consumed")


def save_figure(fig, name: str) -> Path:
    """Write a matplotlib figure to reports/figures and return its path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _benchmark_runs(emissions: pd.DataFrame) -> pd.DataFrame:
    """Return the latest emissions record for each benchmark model in report order."""
    missing_columns = set(REQUIRED_COLUMNS) - set(emissions.columns)
    if missing_columns:
        raise ValueError(f"Emissions data is missing required columns: {', '.join(sorted(missing_columns))}")

    runs = emissions.loc[emissions["project_name"].isin(BENCHMARK_MODELS)].drop_duplicates("project_name", keep="last")
    runs = runs.set_index("project_name").reindex(BENCHMARK_MODELS)
    if runs[["duration", "energy_consumed"]].isna().any().any():
        missing_models = runs.index[runs[["duration", "energy_consumed"]].isna().any(axis=1)].tolist()
        raise ValueError(f"Emissions data is missing benchmark runs or measurements for: {', '.join(missing_models)}")
    return runs


def plot_energy_consumed(emissions: pd.DataFrame) -> Path:
    """Save a bar chart of energy consumed by each benchmark model."""
    runs = _benchmark_runs(emissions)
    figure, axis = plt.subplots(figsize=(9, 5))
    colors = ["#3A7D78", "#3A7D78", "#3A7D78", "#D05A47"]
    bars = axis.bar(runs.index, runs["energy_consumed"], color=colors, zorder=3)
    axis.set_title("Energy consumed per model")
    axis.set_xlabel("Model")
    axis.set_ylabel("Energy consumed (kWh)")
    axis.tick_params(axis="x", labelrotation=15)
    axis.grid(axis="y", color="#D9E1DF", linewidth=0.8, zorder=0)
    axis.bar_label(bars, labels=[f"{value:.6f}" for value in runs["energy_consumed"]], padding=3, fontsize=8)
    figure.tight_layout()
    path = save_figure(figure, "energy_consumed_per_model")
    plt.close(figure)
    return path


def plot_duration_vs_energy(emissions: pd.DataFrame) -> Path:
    """Save a scatter plot relating model duration to energy consumed."""
    runs = _benchmark_runs(emissions)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(runs["duration"], runs["energy_consumed"], color="#3A7D78", s=65, zorder=3)
    label_offsets = {
        "logistic_regression": (8, 8),
        "hist_gradient_boosting": (8, 18),
        "ensemble_soft": (8, -18),
        "ensemble_stacking": (-8, -16),
    }
    for model, run in runs.iterrows():
        axis.annotate(
            model,
            (run["duration"], run["energy_consumed"]),
            xytext=label_offsets[model],
            textcoords="offset points",
            ha="right" if model == "ensemble_stacking" else "left",
        )
    axis.set_title("Duration vs energy consumed")
    axis.set_xlabel("Duration (s)")
    axis.set_ylabel("Energy consumed (kWh)")
    axis.grid(color="#D9E1DF", linewidth=0.8, zorder=0)
    figure.tight_layout()
    path = save_figure(figure, "duration_vs_energy")
    plt.close(figure)
    return path


def generate_sustainability_plots(csv_path: Path = EMISSIONS_PATH) -> tuple[Path, Path]:
    """Generate both sustainability charts from CodeCarbon's emissions log."""
    emissions = pd.read_csv(csv_path)
    return plot_energy_consumed(emissions), plot_duration_vs_energy(emissions)


def main() -> None:
    """Generate both charts and print their output paths."""
    for path in generate_sustainability_plots():
        print(path)


if __name__ == "__main__":
    main()
