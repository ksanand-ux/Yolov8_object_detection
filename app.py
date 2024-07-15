import logging
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_caching import Cache
from PIL import Image

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

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

        # Perform your model inference here
        # For example, replace the following line with your YOLO model inference
        results = model(image)

        # Return the results (example)
        return jsonify({'result': 'success'})

    except Exception as e:
        app.logger.error(f'Unexpected error: {e}', exc_info=True)
        return jsonify({'error': 'Unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
