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
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy handler ---
COPY src/handler.py .

# --- Verify installations and imports ---
RUN python -c "import runpod; print('✅ RunPod version:', runpod.__version__)" && \
    python -c "import torch; print('✅ PyTorch version:', torch.__version__)" && \
    python -c "import iopaint; print('✅ IOPaint imported')" && \
    python -c "from iopaint.model.lama import LaMa; print('✅ LaMa imported')" && \
    python -c "from iopaint.schema import InpaintRequest; print('✅ InpaintRequest imported')" && \
    python -c "import iopaint.model.lama; print('✅ Available in lama module:', [x for x in dir(iopaint.model.lama) if not x.startswith('_')])" && \
    python -c "import iopaint.model; print('✅ Available models:', [x for x in dir(iopaint.model) if not x.startswith('_')])" && \
    pip list | grep -E "(runpod|iopaint|torch)"

# --- Entrypoint for RunPod Serverless ---
CMD ["python", "-u", "handler.py"]