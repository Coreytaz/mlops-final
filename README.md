# End-to-End MLOps Pipeline

Проект реализует полный цикл ML-разработки: от обучения AutoML до A/B тестирования в API.

## Структура проекта

- `dags/` - DAG файлы для Apache Airflow (мониторинг дрифта).
- `models/` - Скрипты обучения (AutoML PyCaret) и артефакты моделей.
- `app/` - Flask REST API с Swagger документацией и логикой A/B тестирования.
- `config/` - Глобальные настройки.
- `logs/` - Логи запросов и метрик.
- `docker-compose.yml` - Конфигурация Docker окружения.

## Быстрый старт с Docker (Рекомендуется)

Все компоненты (API, Airflow, MLflow, Postgres) упакованы в контейнеры.

1. **Запустите проект:**
   ```bash
   docker-compose up -d
   ```
   *Примечание: Первый запуск может занять некоторое время для сборки образов и инициализации базы данных.*

2. **Доступные сервисы:**
   - **Flask API + Swagger**: [http://localhost:5000/apidocs/](http://localhost:5000/apidocs/)
   - **MLflow UI**: [http://localhost:5001](http://localhost:5001)
   - **Airflow Webserver**: [http://localhost:8080](http://localhost:8080)
     - Логин: `admin`
     - Пароль: `admin`

3. **Работа с системой:**
   - **Airflow**: Зайдите в интерфейс, включите DAG `drift_monitoring_automated`. Он будет проверять дрифт каждый день. Можно запустить вручную (Play button).
   - **API**: Используйте Swagger UI для отправки запросов.
   - **AutoML**: Airflow автоматически запустит обучение при дрифте, либо вы можете запустить его вручную внутри контейнера:
     ```bash
     docker exec -it mlops-api python models/automl_training.py
     ```

## Локальная Установка (Без Docker)

1. **Создайте окружение и установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

## Запуск

### 1. Обучение модели (AutoML)
Запустите скрипт обучения. Он скачает датасет Titanic, обучит несколько моделей, выберет лучшую и сохранит её:
```bash
python models/automl_training.py
```
После запуска вы увидите папку `mlruns/` (MLflow эксперименты) и файл `models/best_model_pipeline.pkl`.

### 2. Запуск API (Flask + Swagger)
```bash
python app/api.py
```
Сервер запустится на `http://localhost:5000`.

### 3. Тестирование и Документация
Откройте в браузере: **http://localhost:5000/apidocs/**
Здесь вы увидите Swagger UI, где можно отправить тестовый POST запрос на `/predict` (нажмите "Try it out").

### 4. A/B Тестирование
API автоматически делит трафик 50/50 между "версией A" и "версией B" (в коде сейчас загружается одна и та же модель для демонстрации, но логика реализована).
Результаты всех запросов пишутся в `logs/requests.csv` для дальнейшего анализа.

### 5. Airflow (Мониторинг)
Для запуска DAG требуется установленный Airflow.
DAG лежит в `dags/drift_monitoring_dag.py`.
Он симулирует проверку Drift (PSI) и при превышении порога вызывает скрипт переобучения.

#### Запуск тестов
```bash
python -m unittest tests/test_api.py
```
