import os
import io
import logging
import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import AI Libraries
import tensorflow as tf

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wasteguard")

# --- Konfigurasi Path Model ---
YOLO_MODEL_PATH = "models/yolov8n.pt"
CLASSIFIER_FILE = "models/model_b3_final.h5"

model_yolo = None
model_classifier = None


def load_all_models():
    global model_yolo, model_classifier
    try:
        from ultralytics import YOLO
        base_dir = os.path.dirname(__file__)
        path_yolo       = os.path.join(base_dir, YOLO_MODEL_PATH)
        path_classifier = os.path.join(base_dir, CLASSIFIER_FILE)

        # 1. Load YOLO (Tahap 1 — deteksi objek)
        model_yolo = YOLO(path_yolo)
        logger.info("✅ YOLO loaded")

        # 2. Load model lengkap dari .h5 (arsitektur + weights sekaligus)
        #    CATATAN: file .h5 ini menyimpan model LENGKAP (bukan hanya weights),
        #    jadi HARUS pakai load_model(), bukan load_weights().
        if not os.path.exists(path_classifier):
            logger.error(f"❌ File model tidak ditemukan: {path_classifier}")
            return

        # ── Solusi version mismatch Keras ──────────────────────────────────────
        # File .h5 disimpan dengan Keras versi baru yang menyimpan field
        # 'quantization_config' di Dense config. Keras lama di venv tidak
        # mengenalnya → error saat load_model().
        # Fix: subclass Dense yang membuang field tak dikenal sebelum super().__init__
        class CompatibleDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        model_classifier = tf.keras.models.load_model(
            path_classifier,
            compile=False,
            custom_objects={"Dense": CompatibleDense},
        )

        # Pemanasan model
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model_classifier.predict(dummy, verbose=0)
        logger.info("✅ Classifier (model_b3_final.h5) berhasil dimuat!")
        model_classifier.summary(print_fn=logger.info)

    except Exception as e:
        logger.exception(f"❌ Gagal memuat model: {e}")


# --- Lifespan (pengganti @app.on_event yang sudah deprecated) ---
@asynccontextmanager
async def lifespan(app_instance):
    load_all_models()
    yield

app = FastAPI(title="WasteGuard Two-Stage Binary API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_waste(img_crop: Image.Image):
    """
    Klasifikasi crop gambar: B3 atau Non-B3.

    Label kelas dari notebook (image_dataset_from_directory urutan alfabet):
        index 0 = "B3"      → label biner = 0.0
        index 1 = "non-B3"  → label biner = 1.0

    Jadi: sigmoid output mendekati 0.0 → B3, mendekati 1.0 → Non-B3.
    """
    img = img_crop.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocessing wajib sama dengan saat training
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Prediksi (nilai antara 0.0 dan 1.0)
    prediction = float(model_classifier.predict(img_array, verbose=0)[0][0])

    if prediction < 0.5:
        label    = "Sampah B3"
        category = "B3"
        conf     = round(1.0 - prediction, 4)
    else:
        label    = "Sampah Biasa"
        category = "Non-B3"
        conf     = round(prediction, 4)

    return label, category, conf


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "yolo_loaded":       model_yolo is not None,
        "classifier_loaded": model_classifier is not None,
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    if model_yolo is None or model_classifier is None:
        raise HTTPException(status_code=503, detail="Model belum siap. Cek log server.")

    contents = await image.read()
    pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")

    # Tahap 1: Deteksi lokasi objek dengan YOLO
    results = model_yolo(np.array(pil_img), conf=0.25, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = pil_img.crop((x1, y1, x2, y2))

            # Tahap 2: Klasifikasi B3 / Non-B3 pada setiap crop
            label, category, confidence = predict_waste(crop)

            detections.append({
                "label":      label,
                "category":   category,
                "confidence": confidence,
                "bbox":       [x1, y1, x2, y2],
            })

    return {
        "success":    True,
        "total":      len(detections),
        "detections": detections,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)