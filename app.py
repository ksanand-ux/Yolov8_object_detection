import io

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Load the model

@app.route('/')
def index():
    return "Welcome to the YOLOv8 Object Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        
        # Load the image
        img = Image.open(file.stream)
        img_np = np.array(img)
        
        # Perform prediction
        results = model(img_np)
        
        # Draw bounding boxes on the image
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert image back to PIL format
        img_out = Image.fromarray(img_np)
        img_io = io.BytesIO()
        img_out.save(img_io, 'JPEG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
