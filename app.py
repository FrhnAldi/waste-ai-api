import os
import io
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wasteguard")

YOLO_MODEL_PATH = "yolov8n.pt"
CLASSIFIER_FILE = "model_b3_final.h5"

model_yolo = None
model_classifier = None
models_ready = False

def load_all_models():
    global model_yolo, model_classifier, models_ready
    try:
        import tensorflow as tf
        from ultralytics import YOLO

        base_dir = os.path.dirname(__file__)

        # Load YOLO
        model_yolo = YOLO(os.path.join(base_dir, YOLO_MODEL_PATH))
        logger.info("✅ YOLO loaded")

        # Load Classifier
        path_classifier = os.path.join(base_dir, CLASSIFIER_FILE)
        if not os.path.exists(path_classifier):
            logger.error(f"❌ File tidak ditemukan: {path_classifier}")
            return

        class CompatibleDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        model_classifier = tf.keras.models.load_model(
            path_classifier,
            compile=False,
            custom_objects={"Dense": CompatibleDense},
        )

        # Warmup
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model_classifier.predict(dummy, verbose=0)

        models_ready = True
        logger.info("✅ Semua model siap!")

    except Exception as e:
        logger.exception(f"❌ Gagal memuat model: {e}")

# Jalankan loading di background thread agar port langsung terbuka
threading.Thread(target=load_all_models, daemon=True).start()

app = FastAPI(title="WasteGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_waste(img_crop: Image.Image):
    import tensorflow as tf
    img = img_crop.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    prediction = float(model_classifier.predict(img_array, verbose=0)[0][0])

    if prediction < 0.5:
        return "Sampah B3", "B3", round(1.0 - prediction, 4)
    else:
        return "Sampah Biasa", "Non-B3", round(prediction, 4)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_ready": models_ready,
    }

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    if not models_ready:
        raise HTTPException(status_code=503, detail="Model masih loading, coba lagi dalam 30 detik.")

    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model_yolo(np.array(pil_img), conf=0.25, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = pil_img.crop((x1, y1, x2, y2))
            label, category, confidence = predict_waste(crop)
            detections.append({
                "label": label,
                "category": category,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            })

    return {
        "success": True,
        "total": len(detections),
        "detections": detections,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))