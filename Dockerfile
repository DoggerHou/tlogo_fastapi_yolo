# server/Dockerfile
FROM python:3.10-slim

# Библиотеки для чтения/рисования изображений (Pillow/Ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Установим зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Кладём код приложения
COPY app ./app
# и папку с весами
COPY models ./models

# ENV
ENV MODEL_PATH=models/best.pt
ENV DEVICE=cpu
ENV CONF=0.25
ENV IMGSZ=832
ENV MAX_DETS=300

EXPOSE 8000
# Запускаем FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
