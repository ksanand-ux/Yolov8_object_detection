# Use the official lightweight Python image.
FROM python:3.8-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nginx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask \
    flask_caching \
    flask_executor \
    ultralytics \
    torch \
    pillow \
    opencv-python-headless \
    prometheus_flask_exporter \
    pyjwt \
    gunicorn

# Remove default nginx configuration
RUN rm /etc/nginx/sites-enabled/default

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d

# Expose port 5000
EXPOSE 5000

# Run the web service on container startup.
CMD service nginx start && gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
