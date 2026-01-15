# final-2/mlops-final/dags/model_retraining.py

from __future__ import annotations

import json
from datetime import datetime
import os

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

import mlflow
from mlflow.tracking import MlflowClient

from pycaret.classification import (
    compare_models,
    finalize_model,
    predict_model,
    setup,
)

import requests

from mlflow_pycaret_utils import _ensure_mlflow_experiment_active, _patch_pycaret_mlflow_logger_for_mlflow3


CURRENT_CSV = os.environ.get("DRIFT_CURRENT_PATH", "/opt/airflow/dataset/titanic_current.csv")
TARGET_COL = os.environ.get("TARGET_COL", "Survived")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "titanic_retraining")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "titanic_survival_model")
MLFLOW_MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "Staging")
MLFLOW_PRODUCTION_STAGE = os.environ.get("MLFLOW_PRODUCTION_STAGE", "Production")
PYCARET_SORT_METRIC = os.environ.get("PYCARET_SORT_METRIC", "Accuracy")
MIN_TRAIN_ACCURACY = float(os.environ.get("MIN_TRAIN_ACCURACY", "0.9"))

AB_TEST_CSV = os.environ.get("AB_TEST_CSV", "/opt/airflow/dataset/test.csv")
AB_API_BASE_URL = os.environ.get("AB_API_BASE_URL", "http://api:5000")
AB_REQUESTS = int(os.environ.get("AB_REQUESTS", "100"))
AB_TIMEOUT_SECONDS = float(os.environ.get("AB_TIMEOUT_SECONDS", "15"))


def _train_and_register(**context) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _ensure_mlflow_experiment_active(experiment_name=MLFLOW_EXPERIMENT_NAME, tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = pd.read_csv(CURRENT_CSV)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} not found in dataset: {CURRENT_CSV}")

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

    with mlflow.start_run(run_name="register_best_model") as run:
        mlflow.log_param("dataset_path", CURRENT_CSV)
        mlflow.log_param("target", TARGET_COL)
        mlflow.log_param("rows", int(len(df)))
        mlflow.log_param("cols", int(df.shape[1]))
        mlflow.log_param("pycaret_sort_metric", PYCARET_SORT_METRIC)
        mlflow.log_param("candidate_models", "lr,rf,xgboost,lightgbm")
        mlflow.log_param("best_estimator", type(best).__name__)
        mlflow.log_metric("train_accuracy", acc)
        mlflow.log_param("min_train_accuracy_gate", MIN_TRAIN_ACCURACY)

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

    # NEW: пишем accuracy в description версии модели
    if this_version is not None:
        desc = (
            f"Auto-retrained on {CURRENT_CSV}; "
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
            # чтобы DAG не падал только из-за невозможности обновить description
            pass

    if this_version is not None and MLFLOW_MODEL_STAGE:
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=str(this_version),
            stage=MLFLOW_MODEL_STAGE,
            archive_existing_versions=True,
        )

    ti = context["ti"]
    ti.xcom_push(key="registered_model_name", value=MLFLOW_MODEL_NAME)
    ti.xcom_push(key="registered_model_version", value=this_version)
    ti.xcom_push(key="registered_model_stage", value=MLFLOW_MODEL_STAGE)

    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": this_version,
        "stage": MLFLOW_MODEL_STAGE,
        "train_accuracy": acc,
    }


def _to_jsonable(v):
    if v is None:
        return None

    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        fv = float(v)
        if not np.isfinite(fv):
            return None
        return fv
    if isinstance(v, (np.bool_,)):
        return bool(v)

    if isinstance(v, float):
        if not (v == v and v not in (float("inf"), float("-inf"))):
            return None
        return v
    if isinstance(v, (int, bool, str)):
        return v

    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()

    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]

    return str(v)


def _safe_to_int(x):
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int,)):
            return int(x)
        if isinstance(x, (float,)):
            return int(round(x))
        if isinstance(x, str) and x.strip() != "":
            return int(float(x))
    except Exception:
        return None
    return None


def _run_ab_test_and_decide(**context) -> dict:
    ti = context["ti"]
    model_name = ti.xcom_pull(task_ids="train_and_register", key="registered_model_name")
    model_version = ti.xcom_pull(task_ids="train_and_register", key="registered_model_version")
    if not model_name or not model_version:
        raise ValueError("Missing registered model info in XCom (train_and_register).")

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    prod = client.get_latest_versions(model_name, stages=[MLFLOW_PRODUCTION_STAGE])
    stag = client.get_latest_versions(model_name, stages=[MLFLOW_MODEL_STAGE])

    if not prod or not stag:
        result = {
            "status": "skipped",
            "reason": "no_model_to_compare",
            "has_production": bool(prod),
            "has_staging": bool(stag),
            "model_name": model_name,
            "model_version": int(model_version),
        }
        ti.xcom_push(key="ab_test_result", value=result)
        return result

    df = pd.read_csv(AB_TEST_CSV)
    if len(df) == 0:
        raise ValueError(f"AB test dataset is empty: {AB_TEST_CSV}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"AB test dataset must contain label column {TARGET_COL}: {AB_TEST_CSV}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    if not feature_cols:
        raise ValueError("No feature columns found for AB test payload.")

    url = f"{AB_API_BASE_URL.rstrip('/')}/api/ab/predict"

    n = min(AB_REQUESTS, len(df))
    stats = {"A": {"n": 0, "correct": 0}, "B": {"n": 0, "correct": 0}}
    ok = 0
    failed = 0

    for i in range(n):
        row = df.iloc[i]
        raw_features = {c: row[c] for c in feature_cols}
        features = _to_jsonable(raw_features)
        label = row[TARGET_COL]
        payload = {"user_id": _to_jsonable(i + 1), "features": features}

        resp = requests.post(url, json=payload, timeout=AB_TIMEOUT_SECONDS)
        print(payload)
        print(resp.status_code, resp.text)
        if not (200 <= resp.status_code < 300):
            failed += 1
            continue

        data = resp.json()
        variant = data.get("variant")
        pred = data.get("prediction")
        if variant not in ("A", "B"):
            failed += 1
            continue

        pred_i = _safe_to_int(pred)
        label_i = int(label)
        print(pred_i)
        print(label_i)
        if pred_i is None or label_i is None:
            ok += 1
            continue

        stats[variant]["n"] += 1
        if pred_i == label_i:
            stats[variant]["correct"] += 1
        ok += 1

    a_n = stats["A"]["n"]
    b_n = stats["B"]["n"]
    a_acc = (stats["A"]["correct"] / a_n) if a_n > 0 else None
    b_acc = (stats["B"]["correct"] / b_n) if b_n > 0 else None
    print(stats)
    print(a_n)
    print(b_n)
    print(a_acc)
    print(b_acc)
    if a_acc is None or b_acc is None:
        result = {
            "status": "skipped",
            "reason": "not_enough_labeled_predictions",
            "requests_attempted": int(n),
            "requests_ok": int(ok),
            "requests_failed": int(failed),
            "labeled_A": int(a_n),
            "labeled_B": int(b_n),
            "model_name": model_name,
            "model_version": int(model_version),
        }
        ti.xcom_push(key="ab_test_result", value=result)
        return result

    if b_acc > a_acc:
        client.transition_model_version_stage(
            name=model_name,
            version=str(model_version),
            stage=MLFLOW_PRODUCTION_STAGE,
            archive_existing_versions=True,
        )
        action = "promote_to_production"
        final_stage = MLFLOW_PRODUCTION_STAGE
    else:
        client.transition_model_version_stage(
            name=model_name,
            version=str(model_version),
            stage="None",
            archive_existing_versions=False,
        )
        action = "demote_to_none"
        final_stage = "None"

    result = {
        "status": "done",
        "ab_api_url": url,
        "requests_attempted": int(n),
        "requests_ok": int(ok),
        "requests_failed": int(failed),
        "labeled_A": int(a_n),
        "labeled_B": int(b_n),
        "accuracy_A": float(a_acc),
        "accuracy_B": float(b_acc),
        "action": action,
        "model_name": model_name,
        "model_version": int(model_version),
        "final_stage": final_stage,
    }
    ti.xcom_push(key="ab_test_result", value=result)
    return result


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

    ab_test_and_decide = PythonOperator(
        task_id="ab_test_and_decide",
        python_callable=_run_ab_test_and_decide,
    )

    train_and_register >> ab_test_and_decide