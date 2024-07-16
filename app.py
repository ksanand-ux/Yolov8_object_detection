import io
import logging
import os

from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, url_for)
from flask_caching import Cache
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load the YOLO model
model = YOLO('yolov8n.pt')

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

        draw = ImageDraw.Draw(image)
        try:
            font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans-Bold.ttf")
            font = ImageFont.truetype(font_path, 20)
            app.logger.info('Using DejaVuSans-Bold.ttf')
        except IOError:
            try:
                font_path = os.path.join(os.path.dirname(__file__), "LiberationSans-Bold.ttf")
                font = ImageFont.truetype(font_path, 20)
                app.logger.info('Using LiberationSans-Bold.ttf')
            except IOError:
                font = ImageFont.load_default()
                app.logger.info('Using default font')

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{result.names[int(box.cls[0])]} {box.conf[0]:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)  # Thicker outline
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle(text_bbox, fill="red")  # Background for text
                draw.text((x1, y1), label, fill="white", font=font)

        img_io = io.BytesIO()
        image.save(img_io, 'JPEG')
        img_io.seek(0)
        app.logger.info('Image processed successfully')

        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        app.logger.error(f'Unexpected error: {e}', exc_info=True)
        return jsonify({'error': 'Unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
