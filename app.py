import io

import numpy as np
from flask import Flask, jsonify, request
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
        
        # Convert the image to a format supported by YOLO model
        img = np.array(img)
        
        # Perform prediction
        results = model(img)
        
        # Convert results to JSON serializable format
        predictions = results.pandas().xyxy[0].to_dict(orient='records')
        
        return jsonify(predictions)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

