"""Stage 2: validate the processed data with Great Expectations.

Fails the pipeline when the data stops matching what the model expects, so a
bad upstream file is caught before training instead of after deployment.

The suite below covers the label and grouping contract the modelling stages
rely on. Milestone 3 extends it with per-feature expectations.
"""

import json

import great_expectations as gx
import pandas as pd

from taed2_astra.config import REPORTS_DIR, TRAIN_PATH, VALIDATION_PATH, get_logger, load_params

log = get_logger(__name__)


def build_suite(dataset: dict) -> gx.ExpectationSuite:
    """Return the expectations every training set must satisfy."""
    target = dataset["target"]
    suite = gx.ExpectationSuite(name="processed-train")
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=1))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column=target))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=target))
    suite.add_expectation(
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(column=target, value_set=dataset["classes"])
    )
    if dataset.get("group"):
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=dataset["group"]))
    return suite


def validate(df: pd.DataFrame, dataset: dict) -> dict:
    """Run the suite against a dataframe and return the result as a dict."""
    context = gx.get_context(mode="ephemeral")
    batch = (
        context.data_sources.add_pandas("astra")
        .add_dataframe_asset("processed")
        .add_batch_definition_whole_dataframe("all")
        .get_batch(batch_parameters={"dataframe": df})
    )
    return batch.validate(build_suite(dataset)).to_json_dict()


def main() -> None:
    """Validate the training split and persist the report."""
    params = load_params()
    result = validate(pd.read_parquet(TRAIN_PATH), params["dataset"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_PATH, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    if not result["success"]:
        raise ValueError(f"Data validation failed. See {VALIDATION_PATH}")
    log.info("Data validation passed")


if __name__ == "__main__":
    main()
