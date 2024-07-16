import logging
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_caching import Cache
from PIL import Image
from ultralytics import YOLO  # Add the necessary import for YOLO

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Adjust the path to your model file as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file.save(os.path.join('/app/uploads', file.filename))
    return redirect(url_for('predict', filename=file.filename))

@app.route('/predict', methods=['GET', 'POST'])
@cache.cached(timeout=60, query_string=True)
def predict():
    try:
        filename = request.args.get('filename')
        if filename is None:
            app.logger.error('No filename provided')
            return jsonify({'error': 'No filename provided'}), 400

        file_path = os.path.join('/app/uploads', filename)
        if not os.path.exists(file_path):
            app.logger.error(f'File not found: {file_path}')
            return jsonify({'error': 'File not found'}), 404

        image = Image.open(file_path).convert("RGB")
        app.logger.info(f'Processing image: {filename}')

        # Perform model inference
        results = model(image)
        app.logger.info(f'Inference results: {results}')

        # Format results for JSON response
        response_data = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                response_data.append({
                    'label': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'coordinates': box.xyxy[0].tolist()
                })

        # Return the formatted results
        return jsonify({'result': 'success', 'detections': response_data})

    except Exception as e:
        app.logger.error(f'Unexpected error: {e}', exc_info=True)
        return jsonify({'error': 'Unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
