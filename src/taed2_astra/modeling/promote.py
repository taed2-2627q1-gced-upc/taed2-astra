"""Register the shipped model in the MLflow Model Registry and point the champion alias at it.

`make promote` runs this only after the release gates pass, so a registry version means
"this exact models/model.pkl passed the gates". Each version is tagged with the file's MD5,
the same checksum dvc.lock records, which ties the registry entry to one DVC snapshot.
Promoting the same file twice reuses its version instead of registering a duplicate.
"""

import hashlib
import pickle
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from sklearn.base import BaseEstimator

from taed2_astra.config import MODEL_PATH, get_logger, get_tracking_uri, load_params

log = get_logger(__name__)


def file_md5(path: Path) -> str:
    """Return the MD5 of a file, the checksum DVC uses for outputs."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def promote(model: BaseEstimator, md5: str, candidate: str, registry: dict) -> str:
    """Return the registry version holding the model with this MD5, registering it first if needed."""
    client = MlflowClient()
    name, alias = registry["registered_model"], registry["champion_alias"]
    versions = client.search_model_versions(f"name='{name}'")
    version = next((str(v.version) for v in versions if v.tags.get("model_md5") == md5), None)
    if version is None:
        with mlflow.start_run(run_name=f"promote-{candidate}"):
            mlflow.set_tags({"stage": "promote", "candidate": candidate, "model_md5": md5})
            # Same serializer as train.py: MLflow 3's default (skops) rejects make_column_selector.
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        version = str(info.registered_model_version)
        client.set_model_version_tag(name, version, "model_md5", md5)
        client.set_model_version_tag(name, version, "candidate", candidate)
    client.set_registered_model_alias(name, alias, version)
    return version


def main() -> None:
    """Register models/model.pkl and make it the champion."""
    params = load_params()
    mlflow.set_tracking_uri(get_tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with open(MODEL_PATH, "rb") as handle:
        model = pickle.load(handle)
    version = promote(model, file_md5(MODEL_PATH), params["train"]["model"], params["mlflow"])
    log.info(
        "%s version %s is now @%s",
        params["mlflow"]["registered_model"],
        version,
        params["mlflow"]["champion_alias"],
    )


if __name__ == "__main__":
    main()
