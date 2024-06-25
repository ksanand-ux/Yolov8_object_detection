import io
import logging

from flask import Flask, jsonify, request, send_file
from flask_caching import Cache
from flask_executor import Executor
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
executor = Executor(app)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load the YOLO model
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
@cache.cached(timeout=60, query_string=True)
def predict():
    app.logger.info('Prediction request received')
    if 'file' not in request.files:
        app.logger.error('No file part')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    if file:
        app.logger.info(f'Processing file: {file.filename}')
        image = Image.open(file.stream).convert("RGB")
        results = model(image)
        app.logger.info(f'Detection results: {results}')
        # Process and save the image with detections
        result_image = results[0].plot(show=False)  # Ensure no GUI window is opened
        img_io = io.BytesIO()
        result_image.save(img_io, format='JPEG')
        img_io.seek(0)
        app.logger.info('Image processed and sent back')
        return send_file(img_io, mimetype='image/jpeg')
    app.logger.error('File processing error')
    return jsonify({'error': 'File processing error'}), 500

@app.route('/longtask')
def longtask():
    executor.submit(long_running_function)
    return 'Task started!'

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500

def long_running_function():
    import time
    time.sleep(5)
    app.logger.info('Long running function completed.')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
