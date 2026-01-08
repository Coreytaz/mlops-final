import os

# Базовые пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Создаем папки если их нет
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Конфигурация MLflow
MLFLOW_TRACKING_URI = "file://" + os.path.join(BASE_DIR, "mlruns")
EXPERIMENT_NAME = "automl_experiment"

# Конфигурация мониторинга
DRIFT_THRESHOLD_PSI = 0.2

# Конфигурация API
API_HOST = "0.0.0.0"
API_PORT = 5000
