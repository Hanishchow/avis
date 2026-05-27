FROM python:3.11-slim-bookworm

LABEL description="BirdNET Bioacoustic Detection System"
LABEL maintainer="team"

# System dependencies for audio + ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure logs/detections dirs exist at runtime
RUN mkdir -p logs detections

EXPOSE 5000

# Default: start recorder. Override CMD to run dashboard or both.
CMD ["python", "-u", "birdnet_recorder_enhanced.py"]
