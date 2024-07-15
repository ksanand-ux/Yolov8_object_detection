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
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy local code to the container image.
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the minimal SSL configuration file
COPY options-ssl-nginx.conf /etc/letsencrypt/options-ssl-nginx.conf

# Copy self-signed SSL certificates
RUN mkdir -p /etc/letsencrypt/live/e-see.xyz
COPY certs/fullchain.pem /etc/letsencrypt/live/e-see.xyz/fullchain.pem
COPY certs/privkey.pem /etc/letsencrypt/live/e-see.xyz/privkey.pem

# Copy the Diffie-Hellman parameter file
COPY certs/ssl-dhparams.pem /etc/letsencrypt/ssl-dhparams.pem

# Expose the port on which the app runs
EXPOSE 8080

# Run the web service on container startup.
CMD ["sh", "-c", "nginx && gunicorn -b 0.0.0.0:8080 app:app"]
