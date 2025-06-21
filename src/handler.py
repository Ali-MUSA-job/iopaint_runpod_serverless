import sys
import os
import base64
import importlib
from io import BytesIO
from PIL import Image
import torch
import numpy as np

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline
)

# Python path fix
sys.path.insert(0, '/app')
sys.path.insert(0, '/usr/local/lib/python3.10/site-packages')

try:
    import runpod
    print("✅ RunPod imported successfully")
except ImportError as e:
    print(f"❌ RunPod import failed: {e}")
    sys.path.append('/usr/local/lib/python3.10/dist-packages')
    import runpod
    print("✅ RunPod imported from dist-packages")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Model Registries ---------
MODEL_REGISTRY = {
    "lama": ("iopaint.model.lama", "LaMa"),
    "opencv2": ("iopaint.model.opencv2", "OpenCV2"),
    "zits": ("iopaint.model.zits", "ZITS"),
    "mat": ("iopaint.model.mat", "MAT"),
    "fcf": ("iopaint.model.fcf", "FcF"),
    "mi_gan": ("iopaint.model.mi_gan", "MIGAN"),
    "manga": ("iopaint.model.manga", "Manga"),
    "ldm": ("iopaint.model.ldm", "LDM"),
    "brushnet": ("iopaint.model.brushnet.brushnet", "BrushNetModel")
}

DIFFUSERS_PIPELINE_REGISTRY = {
    "instruct_pix2pix": (StableDiffusionInstructPix2PixPipeline, "timbrooks/instruct-pix2pix"),
    "paint_by_example": (StableDiffusionInpaintPipeline, "Fantasy-Studio/Paint-by-Example"),
    "power_paint": (StableDiffusionInpaintPipeline, "Sanster/PowerPaint_v2")
}

# --------- Globals ---------
loaded_models = {}
current_model = None
current_model_name = None

# --------- Config Class for Attribute Access ---------
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def decode_base64_image(b64_string):
    return Image.open(BytesIO(base64.b64decode(b64_string))).convert("RGB")

def encode_base64_image(image_np):
    """
    Fixed version combining the working parts from both handlers
    """
    print(f"🔍 Encode input shape: {image_np.shape}, dtype: {image_np.dtype}")
    print(f"🔍 Encode input range: min={image_np.min()}, max={image_np.max()}")
    
    # Handle different input formats (from current handler - but simplified)
    if image_np.dtype != np.uint8:
        # Normalize to 0-1 range first, then scale to 0-255
        if image_np.max() > 1.0:
            image_np = image_np / image_np.max()  # Normalize to 0-1
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)
    
    # CRITICAL: Restore the RGB channel flipping from old handler
    if image_np.shape[-1] == 3:
        image_np = image_np[..., ::-1]
    
    print(f"🔍 Final processed shape: {image_np.shape}, dtype: {image_np.dtype}")
    print(f"🔍 Final processed range: min={image_np.min()}, max={image_np.max()}")
    
    image = Image.fromarray(image_np)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def normalize_mask(mask_np):
    """
    Simplified mask processing closer to old handler
    """
    print(f"🔍 Original mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
    
    mask_np = np.squeeze(mask_np)
    print(f"🔍 After squeeze: {mask_np.shape}")
    
    # Simple approach from old handler
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    
    print(f"🔍 Final mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
    return mask_np

def switch_model(requested_model):
    global current_model, current_model_name

    if requested_model == current_model_name:
        return

    if requested_model in loaded_models:
        current_model = loaded_models[requested_model]
        current_model_name = requested_model
        print(f"✅ Switched to cached model: {requested_model}")
        return

    if requested_model in MODEL_REGISTRY:
        module_path, class_name = MODEL_REGISTRY[requested_model]
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        model = ModelClass(device=device)
        print(f"✅ Loaded local model: {requested_model}")
    elif requested_model in DIFFUSERS_PIPELINE_REGISTRY:
        PipelineClass, model_id = DIFFUSERS_PIPELINE_REGISTRY[requested_model]
        model = PipelineClass.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model = model.to(device)
        print(f"✅ Loaded diffusers model: {requested_model}")
    else:
        raise ValueError(f"❌ Unknown model '{requested_model}'.")

    loaded_models[requested_model] = model
    current_model = model
    current_model_name = requested_model

def handler(job):
    try:
        job_input = job.get("input", {})
        image_b64 = job_input.get("image")
        mask_b64 = job_input.get("mask")
        requested_model = job_input.get("model", "lama")

        if not image_b64 or not mask_b64:
            return {"error": "Missing 'image' or 'mask' in input."}

        switch_model(requested_model)

        image = decode_base64_image(image_b64)
        mask = decode_base64_image(mask_b64).convert("L")

        image_np = np.array(image)
        mask_np = normalize_mask(np.array(mask))

        print(f"🔍 Input image shape: {image_np.shape}, dtype: {image_np.dtype}")
        print(f"🔍 Input image range: min={image_np.min()}, max={image_np.max()}")
        print(f"🔍 Input mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")

        if image_np.shape[:2] != mask_np.shape[:2]:
            return {
                "error": f"Image and mask size mismatch. Image: {image_np.shape[:2]}, Mask: {mask_np.shape[:2]}"
            }

        # --------- Prepare Config ---------
        prompt = job_input.get("prompt", "")
        example_b64 = job_input.get("example_image", "")
        example_image = decode_base64_image(example_b64) if example_b64 else None

        if current_model_name in DIFFUSERS_PIPELINE_REGISTRY:
            print("🧪 Running diffusers pipeline...")
            generator = torch.manual_seed(job_input.get("sd_seed", 42))
            result = current_model(
                image=image,
                mask_image=mask,
                prompt=prompt or "",
                example_image=example_image if "paint_by_example" in current_model_name else None,
                guidance_scale=job_input.get("sd_guidance_scale", 7.5),
                num_inference_steps=job_input.get("sd_steps", 50),
                generator=generator,
                output_type="np.array"
            ).images[0]
        else:
            print("🧪 Running local model...")
            
            # SIMPLIFIED CONFIG - Only essential parameters to avoid conflicts
            config = Config(
                prompt=prompt,
                
                # Basic SD parameters (safe defaults)
                sd_steps=job_input.get("sd_steps", 50),
                sd_guidance_scale=job_input.get("sd_guidance_scale", 7.5),
                sd_seed=job_input.get("sd_seed", 42),
                sd_strength=job_input.get("sd_strength", 0.75),  # Reduced from 1.0
                sd_num_samples=job_input.get("sd_num_samples", 1),
                
                # CRITICAL: Conservative settings to prevent blank output
                sd_keep_unmasked_area=job_input.get("sd_keep_unmasked_area", True),
                sd_match_histograms=job_input.get("sd_match_histograms", False),
                sd_mask_blur=job_input.get("sd_mask_blur", 0),  # Reduced from 11
                
                # Minimal HD strategy
                hd_strategy=job_input.get("hd_strategy", "Original"),
                
                # Disable potentially problematic features
                use_croper=False,
                enable_interactive_seg=False,
                enable_remove_bg=False,
                enable_anime_seg=False,
                enable_realesrgan=False,
                enable_gfpgan=False,
                enable_restoreformer=False,
                sd_freeu=False,
                sd_lcm_lora=False
            )

            result = current_model(image_np, mask_np, config)
            print(f"🔍 Local model result shape: {result.shape}, dtype: {result.dtype}")
            print(f"🔍 Local model result range: min={result.min()}, max={result.max()}")

        result_b64 = encode_base64_image(result)
        return {"output_image": result_b64}

    except Exception as e:
        print(f"❌ Handler error: {e}")
        return {"error": str(e)}

# RunPod Serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
