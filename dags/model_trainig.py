from __future__ import annotations

from datetime import datetime
import os

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

import mlflow
from mlflow.tracking import MlflowClient

from pycaret.classification import compare_models, finalize_model, predict_model, setup

from mlflow_pycaret_utils import _ensure_mlflow_experiment_active, _patch_pycaret_mlflow_logger_for_mlflow3


REFERENCE_CSV = os.environ.get("DRIFT_REFERENCE_PATH", "/opt/airflow/dataset/titanic_reference.csv")
TARGET_COL = os.environ.get("TARGET_COL", "Survived")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "titanic_reference_training")

MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_REFERENCE_MODEL_NAME", "titanic_survival_model")
MLFLOW_REFERENCE_STAGE = os.environ.get("MLFLOW_REFERENCE_STAGE", "Production")

PYCARET_SORT_METRIC = os.environ.get("PYCARET_SORT_METRIC", "Accuracy")
MIN_TRAIN_ACCURACY = float(os.environ.get("MIN_TRAIN_ACCURACY", "0.0"))


def _train_reference_and_register(**context) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _ensure_mlflow_experiment_active(experiment_name=MLFLOW_EXPERIMENT_NAME, tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = pd.read_csv(REFERENCE_CSV)
    if len(df) == 0:
        raise ValueError(f"Reference dataset is empty: {REFERENCE_CSV}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} not found in dataset: {REFERENCE_CSV}")

    _patch_pycaret_mlflow_logger_for_mlflow3()

    setup(
        data=df,
        target=TARGET_COL,
        session_id=42,
        fold=5,
        verbose=False,
        html=False,
        log_experiment=True,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        log_plots=False,
        log_data=False,
    )

    best = compare_models(
        sort=PYCARET_SORT_METRIC,
        include=["lr", "rf", "xgboost", "lightgbm"],
    )
    final_model = finalize_model(best)

    preds = predict_model(final_model, data=df)
    pred_col = "prediction_label" if "prediction_label" in preds.columns else "Label"
    if pred_col not in preds.columns:
        raise ValueError("Could not compute accuracy: prediction column not found in predict_model output.")
    acc = float((preds[pred_col] == preds[TARGET_COL]).mean())

    if acc <= MIN_TRAIN_ACCURACY:
        raise ValueError(
            f"Training accuracy gate failed: acc={acc} <= min={MIN_TRAIN_ACCURACY}. Model will not be registered."
        )

    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name="train_reference_model") as run:
        mlflow.log_param("dataset_path", REFERENCE_CSV)
        mlflow.log_param("target", TARGET_COL)
        mlflow.log_param("rows", int(len(df)))
        mlflow.log_param("cols", int(df.shape[1]))
        mlflow.log_param("pycaret_sort_metric", PYCARET_SORT_METRIC)
        mlflow.log_param("candidate_models", "lr,rf,xgboost,lightgbm")
        mlflow.log_param("best_estimator", type(best).__name__)
        mlflow.log_metric("train_accuracy", acc)

        model_info = mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        run_id = run.info.run_id
        model_uri = model_info.model_uri

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        client.get_registered_model(MLFLOW_MODEL_NAME)
    except Exception:
        client.create_registered_model(MLFLOW_MODEL_NAME)

    versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
    this_version = None
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        if getattr(v, "run_id", None) == run_id:
            this_version = int(v.version)
            break

    if this_version is not None:
        desc = (
            f"Reference model trained on {REFERENCE_CSV}; "
            f"train_accuracy={acc:.6f}; "
            f"sort_metric={PYCARET_SORT_METRIC}; "
            f"target={TARGET_COL}"
        )
        try:
            client.update_model_version(
                name=MLFLOW_MODEL_NAME,
                version=str(this_version),
                description=desc,
            )
        except Exception:
            pass

    if this_version is not None and MLFLOW_REFERENCE_STAGE:
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=str(this_version),
            stage=MLFLOW_REFERENCE_STAGE,
            archive_existing_versions=True,
        )

    ti = context["ti"]
    ti.xcom_push(key="reference_model_name", value=MLFLOW_MODEL_NAME)
    ti.xcom_push(key="reference_model_version", value=this_version)
    ti.xcom_push(key="reference_model_stage", value=MLFLOW_REFERENCE_STAGE)

    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": this_version,
        "stage": MLFLOW_REFERENCE_STAGE,
        "train_accuracy": acc,
    }


with DAG(
    dag_id="model_training",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    train_reference_and_register = PythonOperator(
        task_id="train_reference_and_register",
        python_callable=_train_reference_and_register,
    )

    train_reference_and_register