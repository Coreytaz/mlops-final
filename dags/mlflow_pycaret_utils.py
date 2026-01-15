from __future__ import annotations

import contextlib

import mlflow
from mlflow.tracking import MlflowClient


def _patch_pycaret_mlflow_logger_for_mlflow3() -> None:
    """Patch PyCaret's MLflow logger to work with MLflow 3 active run stack.

    Safe no-op if patching isn't needed or imports are unavailable.
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
        previous = mlflow.active_run()
        started = False
        try:
            if run is not None and getattr(run, "info", None) is not None:
                run_id = getattr(run.info, "run_id", None)
                if run_id:
                    mlflow.start_run(run_id=run_id)
                    started = True
            yield
        finally:
            if started:
                mlflow.end_run()
            if previous is not None and getattr(previous, "info", None) is not None:
                prev_id = getattr(previous.info, "run_id", None)
                if prev_id:
                    try:
                        mlflow.start_run(run_id=prev_id)
                    except Exception:
                        pass

    pycaret_mlflow_logger.set_active_mlflow_run = set_active_mlflow_run


def _ensure_mlflow_experiment_active(*, experiment_name: str, tracking_uri: str) -> None:
    """Ensure an MLflow experiment exists and is not in deleted lifecycle stage."""

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name)
        return
    if getattr(exp, "lifecycle_stage", None) == "deleted":
        client.restore_experiment(exp.experiment_id)

patch_pycaret_mlflow_logger_for_mlflow3 = _patch_pycaret_mlflow_logger_for_mlflow3
ensure_mlflow_experiment_active = _ensure_mlflow_experiment_active
