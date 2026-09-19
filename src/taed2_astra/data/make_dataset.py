"""Stage 1: read the raw archive and write train/test splits to data/processed.

When params.yaml names a grouping column, the split is made on groups, so no
entity (e.g. a patient) contributes rows to both halves.
"""

import zipfile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from taed2_astra.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEST_PATH,
    TRAIN_PATH,
    get_logger,
    load_params,
)

log = get_logger(__name__)


def _only_csv(bundle: zipfile.ZipFile) -> str:
    """Return the single CSV member of an archive, or fail with a pointer to params.yaml."""
    csvs = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
    if len(csvs) != 1:
        raise ValueError(f"Expected exactly one CSV in the archive, found {csvs}. Set dataset.raw_member.")
    return csvs[0]


def load_raw(dataset: dict) -> pd.DataFrame:
    """Read the CSV inside the raw archive and drop the columns params.yaml marks as noise."""
    archive = RAW_DATA_DIR / dataset["raw_file"]
    with zipfile.ZipFile(archive) as bundle:
        member = dataset.get("raw_member") or _only_csv(bundle)
        with bundle.open(member) as handle:
            df = pd.read_csv(handle)
    # Strict drop: a misspelt column name here is a data assumption worth failing on.
    return df.drop(columns=dataset.get("drop", []))


def split(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the train and test splits, group-disjoint when a group column is configured."""
    prepare, dataset = params["prepare"], params["dataset"]
    group = dataset.get("group")
    if not group:
        train, test = train_test_split(
            df,
            test_size=prepare["test_size"],
            random_state=prepare["random_state"],
            stratify=df[dataset["target"]],
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=prepare["test_size"], random_state=prepare["random_state"])
    train_idx, test_idx = next(splitter.split(df, groups=df[group]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def main() -> None:
    """Split the raw data and persist both halves as parquet."""
    params = load_params()

    df = load_raw(params["dataset"])
    log.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    train, test = split(df, params)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)
    log.info("Wrote %d train / %d test rows", len(train), len(test))


if __name__ == "__main__":
    main()
