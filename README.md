
# MLOps Final

Небольшой учебный MLOps-проект на датасете Titanic:

- Airflow оркестрирует пайплайны (обучение базовой модели, мониторинг дрейфа, переобучение).
- MLflow используется для трекинга экспериментов и реестра моделей.
- MinIO выступает S3-хранилищем артефактов MLflow.
- Flask API предоставляет предсказания и эндпоинт для A/B.

## Запуск через Docker

Требования: установленный Docker + Docker Compose.

Из корня репозитория:

```bash
docker-compose up --build -d
```

Проверить, что сервисы поднялись:

```bash
docker-compose ps
```

Полезные адреса (по умолчанию):

- Airflow UI: http://localhost:8080 (логин/пароль по умолчанию: `airflow` / `airflow`)
- API: http://localhost:5000
- MLflow UI: http://localhost:5001
- MinIO: http://localhost:9001

Остановить окружение:

```bash
docker-compose down
```

Остановить и удалить volume-данные (БД Airflow/MLflow и данные MinIO):

```bash
docker-compose down -v
```

## Где смотреть пайплайны

DAG-и лежат в папке `dags/` и доступны в Airflow UI.
