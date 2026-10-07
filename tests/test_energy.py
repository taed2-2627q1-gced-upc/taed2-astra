"""CodeCarbon tracking and model card reporting. They need no data on disk."""

import pandas as pd
import pytest
import yaml
from codecarbon import EmissionsTracker

from taed2_astra import energy

RUN = pd.Series(
    {
        "project_name": "train-hist_gradient_boosting",
        "duration": 109.4,
        "emissions": 0.000226,
        "energy_consumed": 0.0013,
        "country_name": "Spain",
        "region": "catalonia",
        "cpu_model": "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz",
        "cpu_count": 12,
        "gpu_count": float("nan"),
        "gpu_model": float("nan"),
        "codecarbon_version": "3.0.0",
    }
)
CARD = {"training_type": "pre-training", "optimization_techniques": "none"}


def test_tracker_row_becomes_metrics_and_tags(tmp_path):
    """A start/stop cycle yields duration, emissions and energy, and writes emissions.csv."""
    tracker = EmissionsTracker(project_name="pytest", output_dir=str(tmp_path), log_level="error")
    tracker.start()
    sum(range(200_000))
    tracker.stop()

    metrics, tags = energy.emissions_summary(tracker)
    assert set(metrics) == {"duration_s", "emissions_kg_co2", "energy_kwh"}
    assert all(isinstance(value, float) and value >= 0.0 for value in metrics.values())
    assert {"cc_country", "cc_region", "cc_cpu_model", "cc_cpu_count"} <= set(tags)
    assert pd.read_csv(tmp_path / "emissions.csv")["project_name"].tolist() == ["pytest"]


def test_latest_run_picks_the_newest_row_of_the_project(tmp_path):
    """Re-training appends to emissions.csv; the card must describe the latest training."""
    path = tmp_path / "emissions.csv"
    pd.DataFrame(
        {"project_name": ["train-hgb", "logistic_regression", "train-hgb"], "emissions": [1.0, 2.0, 3.0]}
    ).to_csv(path, index=False)
    assert energy.latest_run("train-hgb", path)["emissions"] == 3.0
    with pytest.raises(LookupError, match="train-other"):
        energy.latest_run("train-other", path)


def test_card_metadata_follows_the_hugging_face_schema():
    """Emissions in grams, energy in kWh, location, hardware and training time, as the card asks."""
    meta = energy.build_card_metadata(RUN, {"pr_auc": 0.0921, "roc_auc": 0.809}, 1_111_984, 992_970, CARD)
    co2 = meta["co2_eq_emissions"]
    assert co2["emissions"] == pytest.approx(0.226)
    assert co2["power_consumption"] == pytest.approx(0.0013)
    assert co2["source"].startswith("CodeCarbon")
    assert co2["training_type"] == "pre-training"
    assert co2["geographical_location"] == "catalonia, Spain"
    assert co2["hardware_used"] == "12 x Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (CPU)"
    assert co2["training_time"] == pytest.approx(109.4)
    assert meta["model_info"]["model_file_size"] == 1_111_984
    assert meta["model_info"]["datasets_size"] == 992_970
    assert {m["metric"] for m in meta["model_info"]["performance_metrics"]} == {"pr_auc", "roc_auc"}


@pytest.mark.parametrize("gpu_model", ["NVIDIA T4", "1 x NVIDIA T4"])
def test_gpu_is_reported_once_when_codecarbon_saw_one(gpu_model):
    """GPU training must show up in hardware_used, counted once: CodeCarbon 3.x already prefixes the count."""
    run = RUN.copy()
    run["gpu_count"], run["gpu_model"] = 1, gpu_model
    hardware = energy.build_card_metadata(run, {}, 1, 1, CARD)["co2_eq_emissions"]["hardware_used"]
    assert hardware.endswith(", 1 x NVIDIA T4 (GPU)")


def test_front_matter_is_added_then_updated_without_touching_the_body(tmp_path):
    """Re-running the stage rewrites the metadata in place and keeps the prose and any other keys."""
    card = tmp_path / "model_card.md"
    card.write_text("# Model Card\n\nProse.\n", encoding="utf-8")
    energy.write_front_matter(card, {"co2_eq_emissions": {"emissions": 1.0}})
    front = yaml.safe_load(card.read_text(encoding="utf-8").split("---\n")[1])
    front["license"] = "mit"
    card.write_text(f"---\n{yaml.safe_dump(front)}---\n\n# Model Card\n\nProse.\n", encoding="utf-8")

    energy.write_front_matter(card, {"co2_eq_emissions": {"emissions": 2.0}})
    text = card.read_text(encoding="utf-8")
    assert yaml.safe_load(text.split("---\n")[1]) == {"co2_eq_emissions": {"emissions": 2.0}, "license": "mit"}
    assert text.endswith("# Model Card\n\nProse.\n")
    assert text.count("---\n") == 2
