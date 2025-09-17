## Структура проекта
├── app/
│ └── main.py # Точка входа, запуск FastAPI-приложения
├── models/
│ └── best.pt # Обученная модель YOLOv11
├── validation_data/ # Папка с валидационными данными
├── docker-compose.yml # Конфигурация для запуска API в Docker Compose
├── Dockerfile # Сборка Docker-образа API
├── requirements.txt # Зависимости Python
├── tlogo_preparation_for_annotation.ipynb # Ноутбук для подготовки/разметки данных
├── tlogo_training.ipynb # Ноутбук обучения модели
