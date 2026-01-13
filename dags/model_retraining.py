from __future__ import annotations

from datetime import datetime
import os
import contextlib

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

import mlflow
from mlflow.tracking import MlflowClient

from pycaret.classification import (
    compare_models,
    finalize_model,
    predict_model,
    setup,
)

CURRENT_CSV = os.environ.get("DRIFT_CURRENT_PATH", "/opt/airflow/dataset/titanic_current.csv")
TARGET_COL = os.environ.get("TARGET_COL", "Survived")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "titanic_retraining")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "titanic_survival_model")
MLFLOW_MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "Staging")
MLFLOW_PRODUCTION_STAGE = os.environ.get("MLFLOW_PRODUCTION_STAGE", "Production")
PYCARET_SORT_METRIC = os.environ.get("PYCARET_SORT_METRIC", "Accuracy")


def _patch_pycaret_mlflow_logger_for_mlflow3() -> None:
    """Fix PyCaret MLflow logger crash with MLflow 3.

    PyCaret (at least some versions) calls `mlflow.tracking.fluent._active_run_stack.copy()`.
    In MLflow 3 this is a ThreadLocalVariable without `.copy()`, causing:
    AttributeError: 'ThreadLocalVariable' object has no attribute 'copy'

    We patch PyCaret's context manager to safely close any active runs without touching
    the internal stack.
    """

    try:
        from mlflow.tracking import fluent as mlflow_fluent
        active_run_stack = getattr(mlflow_fluent, "_active_run_stack", None)
        if active_run_stack is None or hasattr(active_run_stack, "copy"):
            return
    except Exception:
        return

    try:
        from pycaret.loggers import mlflow_logger as pycaret_mlflow_logger
    except Exception:
        return

    @contextlib.contextmanager
    def clean_active_mlflow_run(): 
        while mlflow.active_run() is not None:
            mlflow.end_run()
        try:
            yield
        finally:
            while mlflow.active_run() is not None:
                mlflow.end_run()

    pycaret_mlflow_logger.clean_active_mlflow_run = clean_active_mlflow_run

    @contextlib.contextmanager
    def set_active_mlflow_run(run):
        """MLflow-3-safe replacement for PyCaret's set_active_mlflow_run().

        PyCaret's implementation uses mlflow internal `_active_run_stack.append/pop`.
        In MLflow 3 this stack is no longer list-like.

        We instead (best-effort) resume the given run via public MLflow APIs.
        """

        previous = mlflow.active_run()
        started = False

        try:
            if run is not None:
                run_id = getattr(getattr(run, "info", None), "run_id", None)
                if run_id:
                    if previous is None or previous.info.run_id != run_id:
                        # Avoid nested runs from leftover context
                        while mlflow.active_run() is not None:
                            mlflow.end_run()
                        mlflow.start_run(run_id=run_id)
                        started = True
            yield
        finally:
            if started and mlflow.active_run() is not None:
                mlflow.end_run()
            # We intentionally do not restore `previous` to avoid resurrecting stale runs
            # inside Airflow task processes.

    pycaret_mlflow_logger.set_active_mlflow_run = set_active_mlflow_run


def _get_ab_test_passed(**context) -> bool:
    dag_run = context.get("dag_run")
    conf = (getattr(dag_run, "conf", None) or {}) if dag_run else {}
    if "ab_test_passed" in conf:
        return bool(conf.get("ab_test_passed"))

    return os.environ.get("AB_TEST_PASSED", "false").strip().lower() in {"1", "true", "yes", "y"}


def _ensure_mlflow_experiment_active(experiment_name: str) -> None:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name)
        return

    # If an experiment was "deleted" in the UI, MLflow prevents setting it active.
    if getattr(exp, "lifecycle_stage", None) == "deleted":
        client.restore_experiment(exp.experiment_id)


def _train_and_register(**context) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _ensure_mlflow_experiment_active(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = pd.read_csv(CURRENT_CSV)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset: {CURRENT_CSV}")

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
    acc = None
    if pred_col in preds.columns:
        acc = float((preds[pred_col] == preds[TARGET_COL]).mean())

    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name="register_best_model") as run:
        mlflow.log_param("dataset_path", CURRENT_CSV)
        mlflow.log_param("target", TARGET_COL)
        mlflow.log_param("rows", int(len(df)))
        mlflow.log_param("cols", int(df.shape[1]))
        mlflow.log_param("pycaret_sort_metric", PYCARET_SORT_METRIC)
        mlflow.log_param("candidate_models", "lr,rf,xgboost,lightgbm")
        mlflow.log_param("best_estimator", type(best).__name__)
        if acc is not None:
            mlflow.log_metric("train_accuracy", acc)

        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
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

    if this_version is not None and MLFLOW_MODEL_STAGE:
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=str(this_version),
            stage=MLFLOW_MODEL_STAGE,
            archive_existing_versions=True,
        )

    ti = context["ti"]
    ti.xcom_push(key="mlflow_run_id", value=run_id)
    ti.xcom_push(key="mlflow_model_uri", value=model_uri)
    ti.xcom_push(key="registered_model_name", value=MLFLOW_MODEL_NAME)
    ti.xcom_push(key="registered_model_version", value=this_version)
    ti.xcom_push(key="registered_model_stage", value=MLFLOW_MODEL_STAGE)

    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": this_version,
        "stage": MLFLOW_MODEL_STAGE,
    }


def _promote_to_production(**context) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    ti = context["ti"]
    model_name = ti.xcom_pull(task_ids="train_and_register", key="registered_model_name")
    model_version = ti.xcom_pull(task_ids="train_and_register", key="registered_model_version")
    if not model_name or not model_version:
        raise ValueError("Missing model_name/model_version in XCom from train_and_register")

    client.transition_model_version_stage(
        name=str(model_name),
        version=str(model_version),
        stage=MLFLOW_PRODUCTION_STAGE,
        archive_existing_versions=True,
    )


with DAG(
    dag_id="model_retraining",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    train_and_register = PythonOperator(
        task_id="train_and_register",
        python_callable=_train_and_register,
    )

    # check_ab_test = ShortCircuitOperator(
    # 	task_id="check_ab_test",
    # 	python_callable=_get_ab_test_passed,
    # )

    # promote_to_production = PythonOperator(
    # 	task_id="promote_to_production",
    # 	python_callable=_promote_to_production,
    # )

    # train_and_register >> check_ab_test >> promote_to_production
    train_and_register

