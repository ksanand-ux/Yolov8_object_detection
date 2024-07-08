import base64
import datetime
import io
import logging
from functools import wraps

import jwt
from flask import Flask, jsonify, request, send_file
from flask_caching import Cache
from flask_cors import CORS
from flask_executor import Executor
from PIL import Image, ImageDraw, ImageFont
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
            image = Image.open(file.stream).convert("RGB")
            app.logger.info(f'Processing image: {file.filename}')
            
            # Perform model prediction
            results = model(image)
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            font = ImageFont.load_default()
            response_metadata = []

            for result in results:
                for box in result.boxes:
                    bbox = box.xyxy[0].tolist()
                    conf = box.conf[0]
                    cls = box.cls[0]
                    label = f"{results.names[int(cls)]} {conf:.2f}"
                    draw.rectangle(bbox, outline="red", width=2)
                    draw.text((bbox[0], bbox[1]), label, fill="white", font=font)
                    response_metadata.append({
                        "label": label,
                        "bbox": bbox,
                        "confidence": conf
                    })

            # Convert image with annotations to bytes
            img_io = io.BytesIO()
            img_with_boxes.save(img_io, 'JPEG')
            img_io.seek(0)

            # Encode the image data in base64
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            app.logger.info('Image processed successfully')
            
            # Return image and metadata
            return jsonify({'image': img_base64, 'detections': response_metadata})
        except Exception as e:
            app.logger.error(f'Error processing image: {e}')
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
    app.run(host='0.0.0.0', port=5000)
