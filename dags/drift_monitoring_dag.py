from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import random

# Эмуляция аргументов по умолчанию
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "drift_monitoring_automated",
    default_args=default_args,
    description="Checks for data drift and triggers retraining if needed",
    schedule_interval=timedelta(days=1),
    catchup=False,
)


def check_data_drift(**kwargs):
    """
    Эмуляция проверки дрифта данных (PSI/KS тест).
    В реальности здесь бы загружались свежие данные и сравнивались с обучающими.
    """
    # Симуляция: генерируем случайный PSI
    psi_score = random.uniform(0.1, 0.4)
    threshold = 0.2

    print(f"Calculated PSI: {psi_score}")

    if psi_score > threshold:
        print("Drift detected! Triggering retraining.")
        return "drift_detected"
    else:
        print("No drift detected.")
        return "no_drift"


def drift_branch_logic(**kwargs):
    """Логика ветвления на основе XCom"""
    ti = kwargs["ti"]
    status = ti.xcom_pull(task_ids="check_drift_task")
    if status == "drift_detected":
        return "trigger_retraining"
    return "skip_retraining"


# Задачи
# 1. Проверка дрифта
check_drift_task = PythonOperator(
    task_id="check_drift_task",
    python_callable=check_data_drift,
    dag=dag,
)

# 2. Ветвление (в Airflow < 2.3 используется BranchPythonOperator, здесь упростим для наглядности)
# Для простоты реализации в "одном файле" сделаем условный вызов внутри Python,
# но "каноничный" способ - это BranchPythonOperator.
# Будем считать, что если дрифт есть, мы идем к trigger_retraining.

trigger_retraining = BashOperator(
    task_id="trigger_retraining",
    # ВАЖНО: Путь к python и скрипту должен быть абсолютным или настроенным в ENV
    bash_command="python /opt/airflow/dags/repo/models/automl_training.py",
    dag=dag,
)

dummy_skip = BashOperator(
    task_id="skip_retraining",
    bash_command='echo "No drift, skipping..."',
    dag=dag,
)

# В простейшем виде без BranchOperator можно использовать следующую логику зависимостей или ShortCircuitOperator
# Но для ТЗ оставим структуру задач.
# Реализация BranchPythonOperator:

from airflow.operators.python import BranchPythonOperator

branching = BranchPythonOperator(
    task_id="branching",
    python_callable=drift_branch_logic,
    provide_context=True,
    dag=dag,
)

check_drift_task >> branching >> [trigger_retraining, dummy_skip]
