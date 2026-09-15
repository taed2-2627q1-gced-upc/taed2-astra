"""Stage 1: read raw data and write the train/test splits to data/processed.

TODO(team): implement load_raw() and the split. Everything the rest of the
pipeline relies on - the params it reads and the two files it must produce -
is already wired.
"""

import pandas as pd

from taed2_astra.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEST_PATH,
    TRAIN_PATH,
    get_logger,
    load_params,
)

log = get_logger(__name__)


def load_raw() -> pd.DataFrame:
    """Read the raw files in data/raw and return them as one dataframe."""
    raise NotImplementedError(f"Read the raw dataset from {RAW_DATA_DIR}")


def split(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the train and test splits."""
    raise NotImplementedError("Split df using params['prepare']")


def main() -> None:
    """Split the raw data and persist both halves as parquet."""
    params = load_params()

    df = load_raw()
    log.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    train, test = split(df, params)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)
    log.info("Wrote %d train / %d test rows", len(train), len(test))


if __name__ == "__main__":
    main()
