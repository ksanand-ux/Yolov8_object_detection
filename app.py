import logging
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO('yolov8n.pt')

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

@app.before_request
def log_request_info():
    app.logger.debug(f'Request Headers: {request.headers}')
    if request.content_type.startswith('multipart/form-data'):
        app.logger.debug('Request Body: Binary data not logged')
    else:
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
    file = request.files['file']
    image_path = os.path.join("data", file.filename)
    file.save(image_path)
    results = model(image_path)
    os.remove(image_path)
    result_image_path = os.path.join("data", "result.jpg")
    results.save(save_dir="data")
    return send_file(result_image_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
