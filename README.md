# Детекция логотипа Т-Банка

## Описание
Решение для задачи автоматического поиска и детекции логотипа **Т-Банка** на изображениях.  
Логотип: стилизованная буква **"Т"** внутри щита (цвет может быть разным).  
API реализовано на **FastAPI**, детекция — с помощью модели **YOLO12s**, обученной на собственном датасете (YOLO format).

## Структура проекта
tlogo_fastapi_yolo/  
├── app/  
│ └── main.py                              # Точка входа, запуск FastAPI-приложения  
├── models/  
│ └── best.pt                              # Обученная модель YOLOv12  
├── validation_data/                       # Папка с валидационными данными  
├── docker-compose.yml                     # Конфигурация для запуска API в Docker Compose  
├── Dockerfile                             # Сборка Docker-образа API  
├── requirements.txt                       # Зависимости Python  
├── tlogo_preparation_for_annotation.ipynb # Ноутбук для подготовки/разметки данных  
└── tlogo_training.ipynb                   # Ноутбук обучения модели  
## API контракт
Эндпоинт `/detect` принимает изображение (JPEG, PNG, BMP, WEBP) и возвращает список найденных логотипов.

```python
class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]
## Пример ответа
{
  "detections": [
    {
      "bbox": {"x_min": 120, "y_min": 85, "x_max": 260, "y_max": 230}
    }
  ]
}
