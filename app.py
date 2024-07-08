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
        return jso
