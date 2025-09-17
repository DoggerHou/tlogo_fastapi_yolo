# FastAPI-инференс для YOLO (CPU/GPU). Без скачиваний — только локальный MODEL_PATH.
import io
import os
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from ultralytics import YOLO

# ---- Настройки через ENV ----
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")  # локальный путь к весам
DEVICE     = os.getenv("DEVICE", "cuda:0")              # "cuda:0" или "cpu"
CONF       = float(os.getenv("CONF", "0.25"))           # минимальный конфид
IMGSZ      = int(os.getenv("IMGSZ", "832"))             # размер инференса
MAX_DETS   = int(os.getenv("MAX_DETS", "300"))          # лимит детекций

# ---- Pydantic модели ----
class BoundingBox(BaseModel):
    x_min: int = Field(..., ge=0)
    y_min: int = Field(..., ge=0)
    x_max: int = Field(..., ge=0)
    y_max: int = Field(..., ge=0)

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

app = FastAPI(title="T-Logo Detector", version="1.0")

# ---- Грузим модель при старте ----
try:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    # пробный прогон — и заодно проверим доступность указанного DEVICE
    try:
        _ = model.predict(source=Image.new("RGB", (32, 32)), device=DEVICE, verbose=False)
    except Exception:
        DEVICE = "cpu"
except Exception as e:
    model = None
    model_init_error = str(e)
else:
    model_init_error = None

def _to_int_xyxy(box, W, H):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

@app.post("/detect", response_model=DetectionResponse, responses={503: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """Детекция логотипа: вернём JSON с боксами (xyxy)."""
    if model is None:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(error="Model not initialized", detail=model_init_error).dict()
        )

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    res = model.predict(
        source=img,
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF,
        max_det=MAX_DETS,
        verbose=False
    )
    r = res[0]
    print(r.boxes)
    H, W = r.orig_shape  # (h, w)

    dets: List[Detection] = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy().tolist()
        conf = r.boxes.conf.cpu().numpy().tolist()
        for box, c in zip(xyxy, conf):
            if float(c) < CONF:
                continue
            x1, y1, x2, y2 = _to_int_xyxy(box, W, H)
            dets.append(Detection(bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)))

    return DetectionResponse(detections=dets)


@app.get("/health")
def health():
    return {"status": "ok"}
