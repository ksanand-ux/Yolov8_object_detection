# Use the official lightweight Python image.
FROM python:3.8-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir flask flask_caching flask_executor ultralytics torch opencv-python-headless pillow

# Run the web service on container startup.
CMD ["python", "app.py"]