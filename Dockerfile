FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- Install Python 3.10 and dependencies ---
RUN apt-get update && \
    apt-get install -y software-properties-common curl git ffmpeg libgl1 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3.10 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# --- Install Python packages ---
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy handler ---
COPY src/handler.py .

# --- Entrypoint for RunPod Serverless ---
CMD ["python3", "-u", "handler.py"]
