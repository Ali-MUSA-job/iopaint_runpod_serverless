import sys
import os
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np

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
    try:
        sys.path.append('/usr/local/lib/python3.10/dist-packages')
        import runpod

        print("✅ RunPod imported from dist-packages")
    except ImportError:
        raise ImportError("Could not import runpod from any location")

from iopaint.model.lama import LaMa
from iopaint.schema import InpaintRequest

# Initialize models globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
    "lama": LaMa(device=device)
}
current_model_name = "lama"
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


# Helper: normalize mask dimensions
def normalize_mask(mask_np):
    """
    Normalize mask to proper shape (H, W) or (H, W, 1)
    """
    print(f"🔍 Original mask shape: {mask_np.shape}")

    # Remove all singleton dimensions first
    mask_np = np.squeeze(mask_np)
    print(f"🔍 After squeeze: {mask_np.shape}")

    # Ensure we have at least 2D
    if mask_np.ndim == 1:
        # This shouldn't happen with proper image masks, but handle it
        raise ValueError(f"Invalid mask shape after processing: {mask_np.shape}")
    elif mask_np.ndim == 2:
        # Perfect - (H, W)
        print(f"✅ Final mask shape: {mask_np.shape}")
        return mask_np
    elif mask_np.ndim == 3:
        # If still 3D after squeeze, take the first channel
        if mask_np.shape[2] == 1:
            mask_np = mask_np[:, :, 0]
        else:
            # Take first channel if multiple channels
            mask_np = mask_np[:, :, 0]
        print(f"✅ Final mask shape: {mask_np.shape}")
        return mask_np
    else:
        raise ValueError(f"Unable to normalize mask with shape: {mask_np.shape}")


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
        mask = decode_base64_image(mask_b64).convert("L")  # single channel

        image_np = np.array(image)
        mask_np = np.array(mask)

        print(f"✅ Original image shape: {image_np.shape}")
        print(f"✅ Original mask shape: {mask_np.shape}")

        # Normalize mask dimensions
        mask_np = normalize_mask(mask_np)

        # Verify shapes are compatible
        if image_np.shape[:2] != mask_np.shape[:2]:
            return {
                "error": f"Image and mask dimensions don't match. Image: {image_np.shape[:2]}, Mask: {mask_np.shape[:2]}"}

        print(f"✅ Final image shape: {image_np.shape}")
        print(f"✅ Final mask shape: {mask_np.shape}")

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
        print(f"❌ Error in handler: {str(e)}")
        return {"error": str(e)}


# RunPod Serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
