FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app:/usr/local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- Install Python 3.10 and dependencies ---
RUN apt-get update && \
    apt-get install -y software-properties-common curl git ffmpeg libgl1 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev python3.10-venv && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3.10 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# --- Copy requirements first for better caching ---
COPY builder/requirements.txt .

# --- Install all packages including RunPod in one go ---
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall runpod>=1.6.0

# --- Copy handler ---
COPY src/handler.py .

# --- Verify installations ---
RUN python -c "import runpod; print('✅ RunPod version:', runpod.__version__)" && \
    python -c "import torch; print('✅ PyTorch version:', torch.__version__)" && \
    python -c "import iopaint; print('✅ IOPaint imported')" && \
    pip list | grep runpod

# --- Entrypoint for RunPod Serverless ---
CMD ["python", "-u", "handler.py"]