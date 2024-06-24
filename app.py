import logging
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    filename='app.log',               # The file where logs will be saved
    level=logging.DEBUG,               # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s %(levelname)s: %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'        # Date format
)

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Load the saved model

@app.before_request
def log_request_info():
    app.logger.debug(f'Request Headers: {request.headers}')
    app.logger.debug(f'Request Body: {request.get_data()}')

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f'Error: {e}')
    return str(e), 500

@app.route('/')
def index():
    app.logger.info('Processing default request')
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
        img_out.save(img_io, format='JPEG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        app.logger.error(f'Error: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
