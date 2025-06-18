import runpod
import os
import base64
from io import BytesIO
from PIL import Image

from iopaint.model_manager import ModelManager

# --- Load ModelManager (shared across requests) ---
model_manager = ModelManager()
loaded_models = {}

# --- Decode/Encode helpers ---
def decode_base64_image(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str)))

def encode_base64_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# --- RunPod Handler ---
def handler(job):
    job_input = job["input"]
    image_b64 = job_input.get("image")
    mask_b64 = job_input.get("mask")
    model_name = job_input.get("model", "lama")  # Default to 'lama'

    if not image_b64 or not mask_b64:
        return {"error": "Missing 'image' or 'mask' in request."}

    try:
        # Decode images
        image = decode_base64_image(image_b64).convert("RGB")
        mask = decode_base64_image(mask_b64).convert("L")

        # Load model if not already loaded
        if model_name not in loaded_models:
            loaded_models[model_name] = model_manager.load_model(model_name)

        model = loaded_models[model_name]

        # Run inpainting
        result = model(image, mask)

        return {"output_image": encode_base64_image(result)}

    except Exception as e:
        return {"error": str(e)}

# --- Start Serverless ---
runpod.serverless.start({"handler": handler})
