"""Stage 2: validate the processed data with Great Expectations.

Fails the pipeline when the data stops matching what the model expects, so a
bad upstream file is caught before training instead of after deployment.

Both splits run the same suite, built from the ``validation`` section of
params.yaml. Expectations are either critical (schema, label, completeness:
the pipeline stops) or warnings (per-feature plausibility ranges: recorded for
review, because a handful of charting errors in clinical data is normal). The
patient-disjointness of the splits is checked here too, since a leak between
them would inflate every metric downstream without any error being raised.
"""

import great_expectations as gx
import pandas as pd
from great_expectations.expectations.metadata_types import FailureSeverity

from taed2_astra.config import (
    REPORTS_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    get_logger,
    load_params,
    write_json,
)

log = get_logger(__name__)

CRITICAL = FailureSeverity.CRITICAL
WARNING = FailureSeverity.WARNING


def expected_columns(dataset: dict, rules: dict) -> list[str]:
    """Return every column a processed split must have: the features, the label and the group."""
    columns = [*rules["ranges"], dataset["target"]]
    if dataset.get("group"):
        columns.append(dataset["group"])
    return columns


def build_suite(dataset: dict, rules: dict) -> gx.ExpectationSuite:
    """Return the expectations every processed split must satisfy."""
    target = dataset["target"]
    low_rate, high_rate = rules["label_rate"]
    suite = gx.ExpectationSuite(name="processed-split")

    def add(expectation: type, severity: FailureSeverity = CRITICAL, **kwargs) -> None:
        suite.add_expectation(expectation(severity=severity, **kwargs))

    # Schema: an added, dropped or renamed column changes what the model is fed.
    add(gx.expectations.ExpectTableRowCountToBeBetween, min_value=rules["min_rows"])
    add(gx.expectations.ExpectTableColumnsToMatchSet, column_set=expected_columns(dataset, rules), exact_match=True)
    for column in expected_columns(dataset, rules):
        add(gx.expectations.ExpectColumnValuesToBeInTypeList, column=column, type_list=rules["dtypes"])

    # Label: complete, in the declared classes, and at a believable prevalence.
    add(gx.expectations.ExpectColumnValuesToNotBeNull, column=target)
    add(gx.expectations.ExpectColumnDistinctValuesToBeInSet, column=target, value_set=dataset["classes"])
    add(gx.expectations.ExpectColumnMeanToBeBetween, column=target, min_value=low_rate, max_value=high_rate)

    if dataset.get("group"):
        add(gx.expectations.ExpectColumnValuesToNotBeNull, column=dataset["group"])
    for column in rules["complete"]:
        add(gx.expectations.ExpectColumnValuesToNotBeNull, column=column)
    for column in rules["binary"]:
        # Nulls are allowed: missingness is a signal the imputer keeps as an indicator.
        add(gx.expectations.ExpectColumnValuesToBeInSet, column=column, value_set=[0, 1])

    for column, (low, high) in rules["ranges"].items():
        add(
            gx.expectations.ExpectColumnValuesToBeBetween,
            WARNING,
            column=column,
            min_value=low,
            max_value=high,
            mostly=rules["range_mostly"],
        )
    return suite


def validate(df: pd.DataFrame, dataset: dict, rules: dict) -> dict:
    """Run the suite against a dataframe and return the Great Expectations result as a dict."""
    context = gx.get_context(mode="ephemeral")
    batch = (
        context.data_sources.add_pandas("astra")
        .add_dataframe_asset("processed")
        .add_batch_definition_whole_dataframe("all")
        .get_batch(batch_parameters={"dataframe": df})
    )
    return batch.validate(build_suite(dataset, rules)).to_json_dict()


def summarise(result: dict) -> list[dict]:
    """Return one compact, diff-friendly row per expectation of a validation result."""
    rows = []
    for item in result["results"]:
        config, outcome = item["expectation_config"], item["result"]
        row = {
            "expectation": config["type"],
            "column": config["kwargs"].get("column"),
            "severity": config.get("severity", CRITICAL.value),
            "success": item["success"],
        }
        if "observed_value" in outcome and not isinstance(outcome["observed_value"], list):
            row["observed_value"] = outcome["observed_value"]
        if outcome.get("unexpected_count"):
            row["unexpected_count"] = outcome["unexpected_count"]
            # Set-level expectations (e.g. distinct values) report a count without a percentage.
            if "unexpected_percent" in outcome:
                row["unexpected_percent"] = round(outcome["unexpected_percent"], 4)
            row["examples"] = outcome.get("partial_unexpected_list", [])[:5]
        rows.append(row)
    return rows


def failures(rows: list[dict], severity: FailureSeverity) -> list[str]:
    """Return a readable label for each failed expectation of the given severity."""
    return [
        f"{row['expectation']}({row['column']})" if row["column"] else row["expectation"]
        for row in rows
        if not row["success"] and row["severity"] == severity.value
    ]


def group_overlap(train: pd.DataFrame, test: pd.DataFrame, group: str) -> int:
    """Return how many groups (e.g. patients) contribute rows to both splits."""
    return len(set(train[group]).intersection(test[group]))


def build_report(splits: dict[str, pd.DataFrame], dataset: dict, rules: dict) -> dict:
    """Validate every split, check they are group-disjoint, and return the combined report."""
    report: dict = {"splits": {}, "critical_failures": [], "warnings": []}
    for name, df in splits.items():
        rows = summarise(validate(df, dataset, rules))
        report["splits"][name] = {"rows": len(df), "expectations": rows}
        report["critical_failures"] += [f"{name}: {label}" for label in failures(rows, CRITICAL)]
        report["warnings"] += [f"{name}: {label}" for label in failures(rows, WARNING)]

    if dataset.get("group") and {"train", "test"} <= splits.keys():
        overlap = group_overlap(splits["train"], splits["test"], dataset["group"])
        report["group_overlap"] = overlap
        if overlap:
            report["critical_failures"].append(f"train/test: {overlap} {dataset['group']} values in both splits")

    report["success"] = not report["critical_failures"]
    return report


def main() -> None:
    """Validate both splits and persist the report; fail on any critical expectation."""
    params = load_params()
    splits = {"train": pd.read_parquet(TRAIN_PATH), "test": pd.read_parquet(TEST_PATH)}
    report = build_report(splits, params["dataset"], params["validation"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(VALIDATION_PATH, report)

    for warning in report["warnings"]:
        log.warning("Data quality warning: %s", warning)
    if not report["success"]:
        raise ValueError(f"Data validation failed: {report['critical_failures']}. See {VALIDATION_PATH}")
    log.info("Data validation passed with %d warning(s)", len(report["warnings"]))


if __name__ == "__main__":
    main()
