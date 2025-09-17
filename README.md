Структура проекта
tlogo_fastapi_yolo/
├── app/                          # папка с кодом FastAPI сервиса
|    └───main.py                  # Точка входа, запуск FastAPI-приложения
├── models/                       
|     └── best.pt                 # Обученная модель YOLOv11
├── validation_data/              # папка с валидационными данными
├── docker-compose.yml            # docker-compose для запуска API
├── Dockerfile                    # сборка образа API
├── requirements.txt              # зависимости Python
├── tlogo_preparation_for_annotation.ipynb   # ноутбук подготовки/разметки
├── tlogo_training.ipynb          # ноутбук обучения модели
