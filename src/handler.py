import sys
import os

# Ensure proper Python path
sys.path.insert(0, '/app')
sys.path.insert(0, '/usr/local/lib/python3.10/site-packages')

try:
    import runpod
    print("✅ RunPod imported successfully")
except ImportError as e:
    print(f"❌ RunPod import failed: {e}")
    print(f"Python path: {sys.path}")
    print(f"Installed packages location: {sys.executable}")
    # Try alternative import locations
    try:
        sys.path.append('/usr/local/lib/python3.10/dist-packages')
        import runpod
        print("✅ RunPod imported from dist-packages")
    except ImportError:
        raise ImportError("Could not import runpod from any location")

import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np

from iopaint.model.lama import LaMa
from iopaint.schema import InpaintRequest

# Initialize models globally on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model instances directly
models = {
    "lama": LaMa(device=device)
}

current_model_name = "lama"  # default
current_model = models[current_model_name]

print(f"✅ Initialized models: {list(models.keys())}")
print(f"✅ Default model: {current_model_name}")


# Helper: decode base64 to PIL image
def decode_base64_image(b64_string):
    return Image.open(BytesIO(base64.b64decode(b64_string))).convert("RGB")


# Helper: encode PIL image to base64 PNG
def encode_base64_image(image_np):
    image = Image.fromarray(image_np)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# Helper: switch model if needed
def switch_model(requested_model):
    global current_model, current_model_name

    if requested_model != current_model_name:
        if requested_model in models:
            current_model = models[requested_model]
            current_model_name = requested_model
            print(f"✅ Switched to model: {requested_model}")
        else:
            raise ValueError(f"Model '{requested_model}' not available. Available models: {list(models.keys())}")


# Main handler
def handler(job):
    try:
        job_input = job.get("input", {})
        image_b64 = job_input.get("image")
        mask_b64 = job_input.get("mask")
        requested_model = job_input.get("model", "lama")

        if not image_b64 or not mask_b64:
            return {"error": "Missing 'image' or 'mask' in input."}

        # Switch model if needed
        switch_model(requested_model)

        # Decode image and mask
        image = decode_base64_image(image_b64)
        mask = decode_base64_image(mask_b64).convert("L")  # mask: 1 channel
        image_np = np.array(image)
        mask_np = np.array(mask)[..., None]  # shape: [H, W, 1]

        # Prepare config
        config = InpaintRequest(
            prompt=job_input.get("prompt", ""),
            enable_controlnet=False,
            enable_brushnet=False,
            enable_powerpaint_v2=False,
            sd_lcm_lora=False
        )

        # Run model directly
        result_np = current_model(image_np, mask_np, config)
        result_b64 = encode_base64_image(result_np)

        return {"output_image": result_b64}

    except Exception as e:
        return {"error": str(e)}


# RunPod Serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
