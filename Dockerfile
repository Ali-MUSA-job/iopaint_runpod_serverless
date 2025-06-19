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


# Entrypoint
CMD ["python3", "-u", "handler.py"]
