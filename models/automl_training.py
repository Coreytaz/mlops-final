import mlflow
import os
import sys
from pycaret.classification import (
    setup,
    compare_models,
    finalize_model,
    save_model,
    pull,
    predict_model,
)
from pycaret.datasets import get_data

# Добавляем путь к корню проекта для импорта конфигов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODELS_DIR


def run_automl_pipeline():
    # 1. Настройка MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Запуск AutoML пайплайна. Tracking URI: {MLFLOW_TRACKING_URI}")

    with mlflow.start_run(run_name="PyCaret_AutoML_Run"):

        # 2. Загрузка данных (используем titanic для примера)
        data = get_data("titanic")

        # 3. Инициализация PyCaret
        # log_experiment=True автоматически логирует метрики в MLflow
        exp_clf = setup(
            data,
            target="Survived",
            session_id=123,
            log_experiment=True,
            experiment_name=EXPERIMENT_NAME,
            html=False,
            verbose=False,
        )

        # 4. Сравнение моделей и выбор лучшей
        print("Сравнение моделей...")
        best_model = compare_models(n_select=1, sort="Accuracy")

        # Получение метрик лучшей модели
        results = pull()
        print(f"Лучшая модель: {best_model}")
        print(results.head())

        # 5. Финализация модели (обучение на всем датасете)
        final_best = finalize_model(best_model)

        # 6. Сохранение модели локально (для A/B тестов и загрузки в API)
        model_path = os.path.join(MODELS_DIR, "best_model_pipeline")
        save_model(final_best, model_path)

        # 7. Регистрация в MLflow Registry
        # PyCaret автоматически логирует артефакты, но мы можем явно зарегистрировать модель
        mlflow.sklearn.log_model(
            final_best, "model", registered_model_name="Production_Candidate"
        )

        print(f"Модель сохранена в {model_path}.pkl и зарегистрирована в MLflow.")


if __name__ == "__main__":
    run_automl_pipeline()
