import io
import logging

from flask import Flask, jsonify, request, send_file
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load YOLO model
model = YOLO('yolov8n.pt')

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        image = Image.open(file.stream).convert('RGB')
        results = model(image)
        # Process and save the image with detections
        result_image = results[0].plot()
        img_io = io.BytesIO()
        result_image.save(img_io, format='JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    return jsonify({'error': 'File processing error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
