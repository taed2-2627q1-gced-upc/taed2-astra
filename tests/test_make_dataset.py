"""Tests for the prepare stage: reading the raw archive and splitting without leakage."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from taed2_astra.data import make_dataset

PREPARE = {"test_size": 0.25, "random_state": 0}


@pytest.fixture(name="frame")
def frame_fixture() -> pd.DataFrame:
    """400 rows from 40 patients with a 10 % positive label."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x": rng.normal(size=400),
            "label": (np.arange(400) % 10 == 0).astype(int),
            "patient": np.arange(400) // 10,
        }
    )


def test_grouped_split_keeps_every_patient_on_one_side(frame):
    """A patient in both halves would leak their own trajectory into the test set."""
    params = {"prepare": PREPARE, "dataset": {"target": "label", "group": "patient"}}
    train, test = make_dataset.split(frame, params)
    assert not set(train["patient"]) & set(test["patient"])
    assert len(train) + len(test) == len(frame)
    assert test["patient"].nunique() == 10  # test_size applies to groups, not rows


def test_split_is_reproducible(frame):
    """Same params, same split: otherwise DVC's cache and the metrics would drift."""
    params = {"prepare": PREPARE, "dataset": {"target": "label", "group": "patient"}}
    first, _ = make_dataset.split(frame, params)
    second, _ = make_dataset.split(frame, params)
    pd.testing.assert_frame_equal(first, second)


def test_ungrouped_split_is_stratified(frame):
    """Without a group column, both halves keep the label prevalence."""
    params = {"prepare": PREPARE, "dataset": {"target": "label", "group": None}}
    train, test = make_dataset.split(frame, params)
    assert train["label"].mean() == pytest.approx(test["label"].mean())


def test_load_raw_reads_the_member_and_drops_noise_columns(frame, tmp_path, monkeypatch):
    """The CSV inside the archive is read and the columns params.yaml marks as noise are removed."""
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Dataset.csv", frame.assign(index_col=0).to_csv(index=False))
    monkeypatch.setattr(make_dataset, "RAW_DATA_DIR", tmp_path)

    df = make_dataset.load_raw({"raw_file": "raw.zip", "raw_member": None, "drop": ["index_col"]})
    pd.testing.assert_frame_equal(df, frame)


def test_load_raw_fails_on_a_misspelt_drop_column(frame, tmp_path, monkeypatch):
    """Dropping a column that does not exist is a stale data assumption, not something to ignore."""
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Dataset.csv", frame.to_csv(index=False))
    monkeypatch.setattr(make_dataset, "RAW_DATA_DIR", tmp_path)

    with pytest.raises(KeyError):
        make_dataset.load_raw({"raw_file": "raw.zip", "drop": ["Unnamed: 0"]})


def test_archive_with_several_csvs_needs_an_explicit_member(tmp_path):
    """Guessing between two CSVs would silently train on the wrong file."""
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("a.csv", "x\n1\n")
        bundle.writestr("b.csv", "x\n2\n")
    with zipfile.ZipFile(archive) as bundle, pytest.raises(ValueError, match="raw_member"):
        make_dataset._only_csv(bundle)  # pylint: disable=protected-access
