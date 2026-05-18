from flask import Flask, request, jsonify
import os

# Import library YOLO/AI kamu di sini
# Contoh: from ultralytics import YOLO

app = Flask(__name__)

# Load Model (Pastikan file model .pt/.h5 ada di folder yang sama)
# model = YOLO('best.pt') 

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"}), 400
    
    file = request.files['image']
    # Simpan sementara atau langsung proses
    file_path = "temp_image.jpg"
    file.save(file_path)

    # --- LOGIKA DETEKSI KAMU ---
    # Contoh format hasil yang diharapkan Laravel:
    results = [
        {
            "label": "Botol Oli", 
            "category": "B3", 
            "confidence": 0.89, 
            "bbox": [100, 200, 50, 50]
        }
    ]
    # ---------------------------

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))