import runpod
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import sys

sys.path.append("/app/iopaint_project")

from iopaint_project.iopaint.model import models
from iopaint_project.iopaint.model.lama import LaMa, AnimeLaMa

# Inject support manually
models["lama"] = LaMa
models["anime-lama"] = AnimeLaMa

from iopaint_project.iopaint.schema import InpaintRequest
from iopaint_project.iopaint import model_manager

# Initialize model manager globally on startup with default model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "lama"  # default
manager = model_manager.ModelManager(name=model_name, device=device)

# Helper: decode base64 to PIL image
def decode_base64_image(b64_string):
    return Image.open(BytesIO(base64.b64decode(b64_string))).convert("RGB")

# Helper: encode PIL image to base64 PNG
def encode_base64_image(image_np):
    image = Image.fromarray(image_np)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

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
        if requested_model != manager.name:
            manager.switch(requested_model)

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

        # Run model
        result_np = manager(image_np, mask_np, config)
        result_b64 = encode_base64_image(result_np)

        return {"output_image": result_b64}

    except Exception as e:
        return {"error": str(e)}

# RunPod Serverless
runpod.serverless.start({"handler": handler})
