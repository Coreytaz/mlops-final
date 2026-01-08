from flask import Flask, request, jsonify
from flasgger import Swagger
import pandas as pd
import sys
import os
import random
import csv
import datetime
from pycaret.classification import load_model, predict_model

# Настройка путей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODELS_DIR, LOGS_DIR

app = Flask(__name__)
swagger = Swagger(app)

# Глобальные переменные для моделей
model_a = None
model_b = None


def load_models():
    """Загрузка моделей при старте"""
    global model_a, model_b
    try:
        # В реальной жизни Model A - это Production, Model B - Staging из MLflow
        # Здесь для демо загрузим одну и ту же модель, но представим их как разные версии
        path = os.path.join(MODELS_DIR, "best_model_pipeline")
        if os.path.exists(path + ".pkl"):
            model_a = load_model(os.path.join(MODELS_DIR, "best_model_pipeline"))
            model_b = load_model(
                os.path.join(MODELS_DIR, "best_model_pipeline")
            )  # В реальности здесь другая версия
            print("Модели загружены успешно.")
        else:
            print(
                "WARN: Модели не найдены. Сначала запустите models/automl_training.py"
            )
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")


def log_request(features, prediction, model_version):
    """Логирование запроса в CSV"""
    log_file = os.path.join(LOGS_DIR, "requests.csv")
    file_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model_version", "prediction", "features"])

        writer.writerow(
            [
                datetime.datetime.now().isoformat(),
                model_version,
                prediction,
                str(features),
            ]
        )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Эндпоинт для предсказания выживаемости (Titanic) с A/B тестированием.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            Pclass:
              type: integer
              example: 3
            Sex:
              type: string
              example: "male"
            Age:
              type: number
              example: 22.0
            SibSp:
              type: integer
              example: 1
            Parch:
              type: integer
              example: 0
            Fare:
              type: number
              example: 7.25
            Embarked:
              type: string
              example: "S"
    responses:
      200:
        description: Результат предсказания
        schema:
          type: object
          properties:
            prediction_label:
              type: integer
            prediction_score:
              type: number
            model_version:
              type: string
    """
    if not model_a:
        return jsonify({"error": "Модели не загружены. Запустите обучение."}), 500

    data = request.get_json()
    df_data = pd.DataFrame([data])

    # A/B логика: 50/50
    if random.random() < 0.5:
        model = model_a
        version = "A (Control)"
    else:
        model = model_b
        version = "B (Test)"

    try:
        # PyCaret predict_model возвращает DataFrame с prediction_label и prediction_score
        prediction = predict_model(model, data=df_data)
        label = int(prediction["prediction_label"].iloc[0])
        score = float(prediction["prediction_score"].iloc[0])

        log_request(data, label, version)

        return jsonify(
            {
                "prediction_label": label,
                "prediction_score": score,
                "model_version": version,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Проверка состояния сервиса"""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
