FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- Install Python 3.10 and dependencies ---
RUN apt-get update && \
    apt-get install -y software-properties-common curl git ffmpeg libgl1 wget && \
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

# --- Clone IOPaint ---
RUN git clone https://github.com/Sanster/IOPaint.git /app/iopaint_project

# --- Set up model directories ---
RUN mkdir -p /app/models /root/.cache/huggingface/hub

# --- Download LaMa model ---
# Method 1: Download directly from HuggingFace
RUN python3 -c "
import torch
from huggingface_hub import hf_hub_download
import os

# Download LaMa model files
model_files = [
    'lama_large.ckpt',
    'config.yaml'
]

# Create model directory
os.makedirs('/app/models/lama', exist_ok=True)

# Download from the official LaMa repository
try:
    for file in model_files:
        print(f'Downloading {file}...')
        # Try different repositories that might have LaMa
        try:
            hf_hub_download(
                repo_id='Sanster/lama-inpainting',
                filename=file,
                local_dir='/app/models/lama',
                local_dir_use_symlinks=False
            )
        except:
            try:
                hf_hub_download(
                    repo_id='wisely134/lama',
                    filename=file,
                    local_dir='/app/models/lama',
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f'Could not download {file}: {e}')
except Exception as e:
    print(f'Error downloading LaMa model: {e}')
"

# --- Alternative: Download using wget (if HuggingFace download fails) ---
RUN cd /app/models && \
    mkdir -p lama && \
    cd lama && \
    wget -q https://github.com/advimman/lama/releases/download/main/big-lama.zip -O big-lama.zip && \
    unzip big-lama.zip && \
    rm big-lama.zip || echo "Direct download failed, trying HuggingFace cache"

# --- Pre-warm the model (optional but recommended) ---
RUN python3 -c "
try:
    import sys
    sys.path.append('/app/iopaint_project')
    from iopaint_project.iopaint import model_manager
    
    # Try to initialize the model to cache it
    print('Pre-warming LaMa model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    manager = model_manager.ModelManager(name='lama', device=device)
    print('LaMa model loaded successfully!')
except Exception as e:
    print(f'Pre-warming failed: {e}')
    print('Model will be downloaded on first use')
"

# --- Set environment variables for model paths ---
ENV IOPAINT_MODEL_DIR=/app/models
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

# --- Copy handler ---
COPY src/handler.py .

# --- Entrypoint for RunPod Serverless ---
CMD ["python3", "-u", "handler.py"]
