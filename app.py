import datetime
import io
import logging
from functools import wraps

import jwt
from flask import Flask, jsonify, request, send_file
from flask_caching import Cache
from flask_cors import CORS
from flask_executor import Executor
from PIL import Image
from prometheus_flask_exporter import PrometheusMetrics
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
executor = Executor(app)
metrics = PrometheusMetrics(app)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load the YOLO model
model = YOLO('yolov8n.pt')

SECRET_KEY = 'your_secret_key'

def encode_auth_token(user_id):
    try:
        payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),
            'iat': datetime.datetime.utcnow(),
            'sub': user_id
        }
        return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    except Exception as e:
        return e

def decode_auth_token(auth_token):
    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=['HS256'])
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return 'Signature expired. Please log in again.'
    except jwt.InvalidTokenError:
        return 'Invalid token. Please log in again.'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        try:
            data = decode_auth_token(token)
        except:
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(*args, **kwargs)
    return decorated

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
    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Open the image file
            image = Image.open(file.stream).convert("RGB")
            app.logger.info(f'Processing image: {file.filename}, Format: {image.format}, Size: {image.size}')
        except Exception as e:
            app.logger.error(f'Error opening file: {e}')
            return jsonify({'error': 'Error opening file'}), 500

        try:
            # Perform model prediction
            results = model(image)
            app.logger.info(f'Model prediction completed successfully, Results: {results}')
        except Exception as e:
            app.logger.error(f'Error during model prediction: {e}')
            return jsonify({'error': 'Error during model prediction'}), 500

        try:
            # Process and encode the result images
            img_bytes = io.BytesIO()
            for result in results:
                result_image = result.plot()  
                result_image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            app.logger.info(f'Result image(s) generated and encoded successfully, size: {img_bytes.getbuffer().nbytes} bytes')
            return send_file(img_bytes, mimetype='image/jpeg')

        except Exception as e:
            app.logger.error(f'Error processing result image(s): {e}')
            return jsonify({'error': 'Error processing result image(s)'}), 500

    app.logger.error('File processing error')
    return jsonify({'error': 'File processing error'}), 500

@app.route('/longtask')
def longtask():
    executor.submit(long_running_function)
    return 'Task started!'

@app.route('/protected', methods=['GET'])
@token_required
def protected():
    return jsonify({'message': 'This is a protected endpoint.'})

@app.errorhandler(404)
def not_found_error(error):
    app.logger.error('404 Not Found: %s', request.url)
    return jsonify({'error': 'Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error('500 Internal Server Error: %s', request.url)
    return jsonify({'error': 'Internal Server Error'}), 500

def long_running_function():
    import time
    time.sleep(5)
    app.logger.info('Long running function completed.')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
