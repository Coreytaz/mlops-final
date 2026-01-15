# final-2/mlops-final/app/api.py
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
import os
import time
import contextlib

import numpy as np
import pandas as pd
from flask import Flask, request
from flask_restx import Api, Resource, fields
from pycaret.datasets import get_data
from evidently import Report
from evidently.presets import DataDriftPreset

import mlflow
from mlflow.tracking import MlflowClient


app = Flask(__name__)
app.config["RESTX_MASK_SWAGGER"] = False

api = Api(
    app,
    version="1.0",
    title="Simple API",
    description="A simple Flask API with Swagger documentation",
    doc="/api/docs",
    prefix="/api",
)

health_ns = api.namespace("health", description="Health check operations")
drift_ns = api.namespace("drift", description="Titanic data drift simulator")

health_model = api.model(
    "Health",
    {
        "status": fields.String(description="Application status"),
    },
)

drift_request = api.model(
    "DriftRequest",
    {
        "severity": fields.Float(description="0..1 intensity of the drift (default 0.35)"),
        "sample_size": fields.Integer(description="How many rows to drift inside the dataset (default 500)"),
    },
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "dataset"
REPORT_DIR = PROJECT_DIR / "reports"
REFERENCE_PATH = DATA_DIR / "titanic_reference.csv"
CURRENT_PATH = DATA_DIR / "titanic_current.csv"
REPORT_PATH = REPORT_DIR / "evidently_report.html"


def ensure_titanic_datasets() -> None:
    """Ensure Titanic reference/current datasets exist on disk (download on API startup if missing)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_PATH.exists():
        df = get_data("titanic", verbose=False)
        df.to_csv(REFERENCE_PATH, index=False)

    if not CURRENT_PATH.exists():
        pd.read_csv(REFERENCE_PATH).to_csv(CURRENT_PATH, index=False)


try:
    ensure_titanic_datasets()
except Exception as exc:
    raise RuntimeError(f"Failed to create Titanic dataset on startup: {exc}") from exc


def apply_drift(df: pd.DataFrame, severity: float) -> pd.DataFrame:
    """Inject controlled drift into Titanic-like data."""
    drift_df = df.copy()
    severity = max(0.0, min(float(severity), 1.0))

    rng = np.random.default_rng()

    if "Age" in drift_df.columns:
        median_age = drift_df["Age"].median()
        noise = rng.normal(loc=10 * severity, scale=5 * severity + 1, size=len(drift_df))
        drift_df["Age"] = (drift_df["Age"].fillna(median_age) + noise).clip(lower=0)

    return drift_df


@health_ns.route("/")
class Health(Resource):
    @health_ns.doc("health_check")
    @health_ns.marshal_with(health_model)
    def get(self):
        """Health check endpoint"""
        return {"status": datetime.now()}, 200


@drift_ns.route("/simulate")
class Drift(Resource):
    @drift_ns.doc("simulate_drift")
    @drift_ns.expect(drift_request, validate=False)
    def post(self):
        """Simulate drift by generating/overwriting titanic_current.csv from titanic_reference.csv."""
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        severity = float(payload.get("severity", 0.35))
        sample_size = int(payload.get("sample_size", 500))

        ensure_titanic_datasets()
        reference_df = pd.read_csv(REFERENCE_PATH)
        if len(reference_df) == 0:
            drift_ns.abort(400, "Reference dataset is empty, cannot simulate drift.")

        current_df = reference_df.copy()
        if sample_size and 0 < sample_size < len(current_df):
            idx = current_df.sample(n=sample_size).index
            drifted_subset = apply_drift(current_df.loc[idx].copy(), severity)
            for col in drifted_subset.columns:
                current_df.loc[idx, col] = drifted_subset[col].values
            modified_rows = int(len(idx))
        else:
            current_df = apply_drift(current_df, severity)
            modified_rows = int(len(current_df))

        current_df.to_csv(CURRENT_PATH, index=False)

        return {
            "message": "Drift simulated and current dataset saved",
            "modified_rows": modified_rows,
            "path": str(CURRENT_PATH),
        }, 200


@drift_ns.route("/report")
class DriftReport(Resource):
    @drift_ns.doc("create_drift_report")
    def post(self):
        """Create Evidently DataDrift HTML report using titanic_reference.csv and titanic_current.csv."""
        sample_size = 500
        random_state = 42
        output_path_raw = str(REPORT_PATH)

        ensure_titanic_datasets()

        reference_df = pd.read_csv(REFERENCE_PATH)
        current_df = pd.read_csv(CURRENT_PATH)

        if len(reference_df) == 0 or len(current_df) == 0:
            drift_ns.abort(400, "Reference/current dataset is empty, cannot build report.")

        if sample_size and sample_size > 0:
            ref_sample = reference_df.sample(n=min(sample_size, len(reference_df)), random_state=random_state)
            cur_sample = current_df.sample(n=min(sample_size, len(current_df)), random_state=random_state)
        else:
            ref_sample = reference_df
            cur_sample = current_df

        report = Report(metrics=[DataDriftPreset()])
        report = report.run(reference_data=ref_sample, current_data=cur_sample)

        output_path = Path(output_path_raw)
        if not output_path.is_absolute():
            output_path = PROJECT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report.save_html(str(output_path))

        return {
            "message": "Data drift report generated",
            "path": str(output_path),
            "reference_rows": int(len(ref_sample)),
            "current_rows": int(len(cur_sample)),
        }, 200


# =========================
# A/B inference via MLflow
# =========================

ab_ns = api.namespace("ab", description="A/B routing and inference via MLflow Registry")

ab_config_model = api.model(
    "ABConfig",
    {
        "model_name": fields.String(description="MLflow registered model name"),
        "tracking_uri": fields.String(description="MLflow Tracking URI"),
        "traffic_b_percent": fields.Integer(description="0..100 traffic percent to Staging (B)"),
        "experiment_name": fields.String(description="MLflow experiment name for inference logs"),
    },
)

ab_set_config_request = api.model(
    "ABSetConfigRequest",
    {
        "traffic_b_percent": fields.Integer(description="0..100 traffic percent to Staging (B)"),
    },
)

ab_predict_request = api.model(
    "ABPredictRequest",
    {
        "user_id": fields.Integer(description="User id for stable routing"),
        "features": fields.Raw(description="Feature dict for model input"),
    },
)

ab_predict_response = api.model(
    "ABPredictResponse",
    {
        "variant": fields.String(description="A or B"),
        "stage": fields.String(description="MLflow stage"),
        "model_version": fields.String(description="Model version in registry"),
        "prediction": fields.Raw(description="Prediction output"),
        "latency_ms": fields.Float(description="Inference latency in ms"),
    },
)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "titanic_survival_model")
MLFLOW_AB_EXPERIMENT = os.environ.get("MLFLOW_AB_EXPERIMENT", "ab_inference")
TRAFFIC_B_PERCENT = int(os.environ.get("TRAFFIC_B_PERCENT", "50"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_ab_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _ensure_mlflow_experiment_active(experiment_name: str) -> None:
    exp = _ab_client.get_experiment_by_name(experiment_name)
    if exp is None:
        _ab_client.create_experiment(experiment_name)
        return
    if getattr(exp, "lifecycle_stage", None) == "deleted":
        _ab_client.restore_experiment(exp.experiment_id)


def _choose_variant(user_id: int) -> str:
    bucket = random.randint(0, 100)
    return "B" if bucket < TRAFFIC_B_PERCENT else "A"


def _stage_for_variant(variant: str) -> str:
    return "Staging" if variant == "B" else "Production"


def _load_model(stage: str):
    latest = _ab_client.get_latest_versions(MLFLOW_MODEL_NAME, stages=[stage])
    if not latest:
        raise ValueError(f"No model versions found in stage={stage}")
    mv = latest[0]
    key = (stage, str(mv.version))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], mv

    model_uri = f"models:/{MLFLOW_MODEL_NAME}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    _MODEL_CACHE[key] = model
    return model, mv


def _to_jsonable(v: Any) -> Any:
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
        if not np.isfinite(v):
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


def _normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    clean = _to_jsonable(features)
    if not isinstance(clean, dict):
        return {}
    return clean


def _sanitize_df_for_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Normalize common "missing" markers to real NaN
    out = out.replace(
        to_replace=[
            None,
            "",
            " ",
            "None",
            "none",
            "NaN",
            "nan",
            "NULL",
            "null",
            "N/A",
            "n/a",
        ],
        value=np.nan,
    )

    # Ensure inf is treated as missing
    out = out.replace([np.inf, -np.inf], np.nan)

    # Try to coerce obvious numeric-looking object columns to numeric where possible
    # (keeps non-numeric as-is)
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = pd.to_numeric(out[col], errors="ignore")

    return out


def _log_ab_event(
    *,
    variant: str,
    stage: str,
    mv: Any,
    features: Dict[str, Any],
    prediction: Any,
    user_id: int,
    latency_ms: float,
) -> None:
    _ensure_mlflow_experiment_active(MLFLOW_AB_EXPERIMENT)
    mlflow.set_experiment(MLFLOW_AB_EXPERIMENT)

    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name="ab_inference_event"):
        mlflow.set_tag("ab_variant", variant)
        mlflow.set_tag("model_name", MLFLOW_MODEL_NAME)
        mlflow.set_tag("model_stage", stage)
        mlflow.set_tag("model_version", str(getattr(mv, "version", "")))
        mlflow.log_param("user_id", int(user_id))
        mlflow.log_metric("latency_ms", float(latency_ms))

        mlflow.log_dict(_to_jsonable(features), "features.json")
        mlflow.log_dict({"prediction": _to_jsonable(prediction)}, "prediction.json")


@ab_ns.route("/config")
class ABConfig(Resource):
    @ab_ns.doc("get_ab_config")
    @ab_ns.marshal_with(ab_config_model)
    def get(self):
        return {
            "model_name": MLFLOW_MODEL_NAME,
            "tracking_uri": MLFLOW_TRACKING_URI,
            "traffic_b_percent": TRAFFIC_B_PERCENT,
            "experiment_name": MLFLOW_AB_EXPERIMENT,
        }, 200

    @ab_ns.doc("set_ab_config")
    @ab_ns.expect(ab_set_config_request, validate=False)
    @ab_ns.marshal_with(ab_config_model)
    def post(self):
        global TRAFFIC_B_PERCENT
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        if "traffic_b_percent" not in payload:
            ab_ns.abort(400, "Missing traffic_b_percent")

        try:
            p_int = int(payload["traffic_b_percent"])
        except Exception:
            ab_ns.abort(400, "traffic_b_percent must be int")

        if p_int < 0 or p_int > 100:
            ab_ns.abort(400, "traffic_b_percent must be in 0..100")

        TRAFFIC_B_PERCENT = p_int

        return {
            "model_name": MLFLOW_MODEL_NAME,
            "tracking_uri": MLFLOW_TRACKING_URI,
            "traffic_b_percent": TRAFFIC_B_PERCENT,
            "experiment_name": MLFLOW_AB_EXPERIMENT,
        }, 200


@ab_ns.route("/predict")
class ABPredict(Resource):
    @ab_ns.doc("ab_predict")
    @ab_ns.expect(ab_predict_request, validate=False)
    @ab_ns.marshal_with(ab_predict_response)
    def post(self):
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        if "user_id" not in payload:
            ab_ns.abort(400, "Missing user_id")
        if "features" not in payload or not isinstance(payload["features"], dict):
            ab_ns.abort(400, "Missing features (object)")

        user_id = int(payload["user_id"])
        features = _normalize_features(payload["features"])

        variant = _choose_variant(user_id)
        stage = _stage_for_variant(variant)

        t0 = time.perf_counter()
        try:
            model, mv = _load_model(stage)
        except Exception as exc:
            ab_ns.abort(500, f"Failed to load model from MLflow: {exc}")

        df = pd.DataFrame([features])
        df = _sanitize_df_for_sklearn(df)

        try:
            y = model.predict(df)
        except Exception as exc:
            ab_ns.abort(400, f"Model prediction failed: {exc}")

        pred: Any
        if isinstance(y, (list, tuple, np.ndarray, pd.Series)):
            pred = y[0] if len(y) > 0 else None
        else:
            pred = y

        pred = _to_jsonable(pred)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        try:
            _log_ab_event(
                variant=variant,
                stage=stage,
                mv=mv,
                features=features,
                prediction=pred,
                user_id=user_id,
                latency_ms=latency_ms,
            )
        except Exception:
            pass

        return {
            "variant": variant,
            "stage": stage,
            "model_version": str(getattr(mv, "version", "")),
            "prediction": pred,
            "latency_ms": float(latency_ms),
        }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)