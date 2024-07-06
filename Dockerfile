# Use the official lightweight Python image.
FROM python:3.8-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nginx

# Set the working directory
WORKDIR /app

# Copy local code to the container image.
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask \
    flask_caching \
    flask_executor \
    flask_cors \
    ultralytics \
    prometheus-flask-exporter \
    pyjwt \
    gunicorn

# Run the web service on container startup.
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
