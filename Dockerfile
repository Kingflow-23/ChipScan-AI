FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy code
COPY app/ ./app/
COPY model/ ./model/

# Expose Flask port
EXPOSE 5000

# Production server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app.app:app"]
