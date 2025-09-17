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
```
## Пример ответа
```
{
  "detections": [
    {
      "bbox": {"x_min": 120, "y_min": 85, "x_max": 260, "y_max": 230}
    }
  ]
}
```
## Запуск
•	Создать пустую папку на ПК  
•	Убедиться, что установлен Docker / установить  
•   Склонировать репозиторий:  
    git clone https://github.com/<your_name>/Fastapi_YOLO.git  
    cd Fastapi_YOLO  
•   Собрать и запустить контейнер:  
    docker compose up --build -d  
•	Обращаться к localhost:8000  


# Проделанные шаги

## Первоначальная разметка набора данных
Полученнный датасет содержит 30079 изображений, ручная разметка такого набора данных силами одного человека (в адекватные сроки) невозможна, поэтому воспользуемся zero-shot подходом для первоначальной разметки изображений. После того, как
получится добиться приемлемого качества на валидационном наборе - прогоним *zero-shot* модель на 2000 изображений из выданного набора. Далее будем вручную корректировать разметку, удаляя ложные срабатывания и добавляя неразмеченные объекты. Исходя из этого целевой метрикой данного этапа будет являться recall, ведь удалять лишние срабатывания куда проще, чем добавлять новые. По итогу цель данного этапа - сократить время на ручную разметку

Поскольку zero-shot моделей для задач object detection много, воспользуемся информацией со страницы https://huggingface.co/models?pipeline_tag=zero-shot-object-detection и возьмем самые популярные архитектуры. В первую очередь обращаем внимание на модели, обновлявшиеся в 2024-2025 году, тут на выбор есть 2 основные архитектуры — OWLiT(OWLv2), Grounding DINO.

Каким образом будет происходить оценка работ всех этих моделей? На портале Roboflow уже находится некоторое количество размеченных изображений (https://universe.roboflow.com/jull-7hwfj/1200-8o5s6), выполнив ручную проверку данного набора убеждаемся, что разметка была проведена верно. Составленный набор содержит большое количество логотипов Т-банка разных цветов, размеров, снятых с разных углов. В нем имеется достаточное количество изображений логотипа банка "Тинькофф", которые следует избегать по условию кейса. Также в набор собственноручно было добавлено некоторое количество изображений, никак не относящихся к "Т-банку", что будет являться хорошей проверкой на False Positive срабатывания.

Имея опыт работы с задачей детекции используя фреймворк Ultralytics и модели YOLO — я проверил как показывает себя модель YOLOv8x-worldv2, однако она показала результаты Precision и Recall ниже 0.1. Поскольку YOLO-world это open-vocabulary детектор, он плохо справляется с обнаружением изображений по длинным фразам, оставим его до лучших времен.


#### Чтобы быстро получить стартовые боксы, обучил **две zero-shot модели**:
- **OWLv2 (google/owlv2-large-patch14-ensemble)** — выше точность, низкая полнота.
- **Grounding DINO (IDEA-Research/grounding-dino-base)** — высокий полнота, низкая точность.  

Далее были попытки найти подходящие промты, дабы повысить качество **zero-shot** детекции. Итоговые варианты:
```
PROMPTS_OWL = [
    "a stylized geometric letter t inside a shield emblem",
    "a minimalist t logo inside a angular shield shape",
    "a modern t letter inside a geometric shield outline",
    "a T-Bank logo with letter t in shield",
    "t letter in white shield"
]
PROMPTS_GDINO = (
    "logo with a bold black letter T inside a white shield-like shape",
    "white or yellowshield emblem with a large black T in the center",
    "white badge shaped like a shield with a black capital T",
    "minimalist logo with a bold T inside a shield icon",
    "flat design emblem shaped like a shield containing a yellow or black or white T letter",
    "logo featuring a strong black T on a yellow or white shield background",
    "simplified yellow or black or white shield logo with a single bold yellow or black or white letter T"
)
```
Как видно, модель GDINO любиь длинные описания, **после четкого указания наиболее используемых цветов(желтый, белый, черный)** recall у GDINO вырос на 0.3.
Для отслеживания того, где модель ошибается была реализована функция, которая сохраняет в папке изображение с его **GT** и **inference** боксами.

## Проведенные эксперименты на данном этапе ##
- **Grayscale** vs **Color**: выяснил, что перевод в Ч/Б **снижает** качество.
Собрал три режима для GDINO:
- **precise** — более жёсткие пороги (Box/Text), лёгкий shape-фильтр (аспект/площадь), NMS, вырос precision.  
- **recall** — мягкие пороги **в два прохода** (union), более широкий shape-фильтр, NMS, вырос recall.  
- **compromise** — union(`precise`,`recall`) + финальный NMS → баланс.
- **NMS**, повышение до 0.6 повысило recall
Далее собрал 3 ансамбля **OWL + GDINO**
- **consensus** — оставляем те OWL-боксы, которые подтверждены GDINO по **IoU≥0.5** , вырос precision.
- **union** — объединяем боксы обеих моделей + финальный NMS, вырос recall.
- **WBF** — простая **Weighted Boxes Fusion** с весами (OWL=1.0, GDINO=0.85), кластеризация по IoU≥0.5 → сглаживание координат.

**Итоговые результаты обучения zero-shot моделей:** 

| run                     | precision | recall   | f1       | mAP@50  | mAP@50-95 | images | conf | iou_eval |
|-------------------------|-----------|----------|----------|---------|-----------|--------|------|----------|
| ENSEMBLE_CONSENSUS      | 0.819672  | 0.416667 | 0.552486 | 0.365200| 0.336693  | 146    | 0.3  | 0.5      |
| OWLv2_COLOR             | 0.732394  | 0.433333 | 0.544503 | 0.374188| 0.344315  | 146    | 0.3  | 0.5      |
| GroundingDINO_PRECISE   | 0.413408  | 0.616667 | 0.494983 | 0.291816| 0.242893  | 146    | 0.3  | 0.5      |
| ENSEMBLE_UNION          | 0.240185  | 0.866667 | 0.376130 | 0.403787| 0.345705  | 146    | 0.3  | 0.5      |
| ENSEMBLE_WBF            | 0.240185  | 0.866667 | 0.376130 | 0.268609| 0.212821  | 146    | 0.3  | 0.5      |
| GroundingDINO_COMPROMISE| 0.238318  | 0.850000 | 0.372263 | 0.353321| 0.287187  | 146    | 0.3  | 0.5      |
| GroundingDINO_RECALL    | 0.236659  | 0.850000 | 0.370236 | 0.353321| 0.287187  | 146    | 0.3  | 0.5      |
| GroundingDINO_BASE      | 0.139869  | 0.891667 | 0.241808 | 0.236896| 0.195611  | 146    | 0.3  | 0.5      |

По итогам, для упрощения дальнейшей разметки была выбрана модель ENSEMBLE_UNION, т.к. она наилучший (в данном случае) баланс precision/recall


