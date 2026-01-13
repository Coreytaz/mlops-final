from __future__ import annotations

from datetime import datetime
import os

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

REFERENCE_CSV = os.environ.get("DRIFT_REFERENCE_PATH", "/opt/airflow/dataset/titanic_reference.csv")
CURRENT_CSV = os.environ.get("DRIFT_CURRENT_PATH", "/opt/airflow/dataset/titanic_current.csv")


PSI_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05

def _get_metrics_list(dict_eval: dict) -> list[dict]:
    metrics = dict_eval.get("metrics", [])
    if not isinstance(metrics, list):
        return []
    return metrics


def _extract_metric_values(dict_eval: dict) -> dict:
    out: dict[str, float] = {}
    for m in _get_metrics_list(dict_eval):
        print('--- metric ---')
        print(m)
        name = m.get("metric_name")
        val = m.get("value")
        if not isinstance(name, str):
            continue

        num_val = None
        if isinstance(val, (int, float)):
            num_val = float(val)
        elif isinstance(val, dict):
            if isinstance(val.get("value"), (int, float)):
                num_val = float(val["value"])
            elif isinstance(val.get("p_value"), (int, float)):
                num_val = float(val["p_value"])

        if num_val is None:
            continue

        out[name] = num_val

    return out

def _compute_drift(**context) -> bool:
    reference_df = pd.read_csv(REFERENCE_CSV)
    current_df = pd.read_csv(CURRENT_CSV)

    X = current_df.drop(columns=["Survived"], errors="ignore")
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

    schema = DataDefinition(
        numerical_columns=numerical_features,
        categorical_columns=categorical_features,
    )

    eval_ref = Dataset.from_pandas(reference_df, data_definition=schema)
    eval_cur = Dataset.from_pandas(current_df, data_definition=schema)

    report = Report([DataDriftPreset()])
    my_eval = report.run(current_data=eval_cur, reference_data=eval_ref)
    dict_eval = my_eval.dict()

    metric_map = _extract_metric_values(dict_eval)

    psi_values = [v for k, v in metric_map.items() if "PSI" in k or "psi" in k.lower()]
    max_psi = max(psi_values) if psi_values else None

    ks_pvalues = [v for k, v in metric_map.items() if "K-S p_value" in k or "K-S" in k]
    ks_drifted = any(p < KS_PVALUE_THRESHOLD for p in ks_pvalues) if ks_pvalues else False
    trigger = False
    if max_psi is not None and max_psi > PSI_THRESHOLD:
        trigger = True
    if ks_drifted:
        trigger = True

    ti = context["ti"]
    ti.xcom_push(key="max_psi", value=max_psi)
    ti.xcom_push(key="ks_drifted", value=ks_drifted)
    ti.xcom_push(key="trigger_retrain", value=trigger)

    return trigger


def _should_trigger_retrain(**context) -> bool:
    ti = context["ti"]
    return bool(ti.xcom_pull(key="trigger_retrain", task_ids="compute_drift"))

with DAG(
    dag_id="data_drift_monitoring",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:
    compute_drift = PythonOperator(
        task_id="compute_drift",
        python_callable=_compute_drift,
    )

    check_trigger = ShortCircuitOperator(
        task_id="check_trigger",
        python_callable=_should_trigger_retrain,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="model_retraining",
        wait_for_completion=False,
    )

    compute_drift >> check_trigger >> trigger_retrain