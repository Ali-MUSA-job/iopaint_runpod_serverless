FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY builder/requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY src/handler.py .

# Clone IOPaint repo
RUN git clone https://github.com/Sanster/IOPaint.git /app/iopaint_project

# Set PYTHONPATH for imports
ENV PYTHONPATH=/app

# Test network access with requests
RUN python3 -c "import requests; print('Network OK:', requests.get('https://github.com').status_code)"

# Manually download LaMa model to cache to avoid python download errors
RUN mkdir -p /root/.cache/iopaint && \
    wget -O /root/.cache/iopaint/big-lama.pt https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt

# Optional: download Anime LaMa model too if needed
RUN wget -O /root/.cache/iopaint/anime-manga-big-lama.pt https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt || true

# Entrypoint
CMD ["python3", "-u", "handler.py"]
