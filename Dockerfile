# Use the official lightweight Python image.
FROM python:3.8-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nginx \
    certbot \
    python3-certbot-nginx

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir flask flask_caching flask_executor ultralytics torch pillow opencv-python-headless prometheus_flask_exporter pyjwt

# Expose port 5000
EXPOSE 5000

# Run the web service on container startup.
CMD ["python", "app.py"]
