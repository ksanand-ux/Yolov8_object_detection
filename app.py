import torch
from flask import Flask, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Load the model

@app.route('/')
def index():
    return "Welcome to the YOLOv8 Object Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    results = model(file.read())
    return jsonify(results.pandas().xyxy[0].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
