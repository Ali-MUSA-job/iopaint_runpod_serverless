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
    "power_paint": (StableDiffusionInpaintPipeline, "Sanster/PowerPaint-V1-stable-diffusion-inpainting")
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
    print(f"🔍 Encode input shape: {image_np.shape}, dtype: {image_np.dtype}")
    print(f"🔍 Encode input range: min={image_np.min()}, max={image_np.max()}")

    # Handle different input formats
    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
        # If values are in 0-1 range, scale to 0-255
        if image_np.max() <= 1.0:
            image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
        # If values are already in 0-255 range but float
        else:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    elif image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    # Don't flip RGB channels - keep original format
    # Remove this line: if image_np.shape[-1] == 3: image_np = image_np[..., ::-1]

    print(f"🔍 Final processed shape: {image_np.shape}, dtype: {image_np.dtype}")
    print(f"🔍 Final processed range: min={image_np.min()}, max={image_np.max()}")

    image = Image.fromarray(image_np)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def normalize_mask(mask_np):
    print(f"🔍 Original mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
    print(f"🔍 Original mask range: min={mask_np.min()}, max={mask_np.max()}")

    mask_np = np.squeeze(mask_np)
    print(f"🔍 After squeeze: {mask_np.shape}")

    # Handle 3-channel masks
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    # Ensure mask values are in correct range (0-255 for uint8)
    if mask_np.dtype == np.uint8:
        # Mask should be 0 for areas to keep, 255 for areas to inpaint
        print(f"🔍 Mask unique values: {np.unique(mask_np)}")
    else:
        # Convert to uint8 if needed
        mask_np = (mask_np * 255).astype(np.uint8)
        print(f"🔍 Converted mask unique values: {np.unique(mask_np)}")

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
            # Create config object with all expected attributes for local models
            config = Config(
                # Basic parameters
                prompt=prompt,

                # CRITICAL: These parameters can cause white/blank outputs
                sd_keep_unmasked_area=job_input.get("sd_keep_unmasked_area", True),
                # Should be True to preserve original
                sd_match_histograms=job_input.get("sd_match_histograms", False),  # Can cause issues if True
                sd_mask_blur=job_input.get("sd_mask_blur", 0),  # Try 0 first, then 11
                sd_strength=job_input.get("sd_strength", 0.75),  # Try lower values like 0.75

                # Stable Diffusion parameters
                sd_steps=job_input.get("sd_steps", 50),
                sd_guidance_scale=job_input.get("sd_guidance_scale", 7.5),
                sd_seed=job_input.get("sd_seed", 42),
                sd_num_samples=job_input.get("sd_num_samples", 1),
                sd_freeu=job_input.get("sd_freeu", False),
                sd_freeu_config=job_input.get("sd_freeu_config", {}),
                sd_lcm_lora=job_input.get("sd_lcm_lora", False),

                # HD Strategy - IMPORTANT: Can cause issues with certain models
                hd_strategy=job_input.get("hd_strategy", "Original"),  # Try "Original" first
                hd_strategy_crop_margin=job_input.get("hd_strategy_crop_margin", 32),
                hd_strategy_crop_trigger_size=job_input.get("hd_strategy_crop_trigger_size", 512),
                hd_strategy_resize_limit=job_input.get("hd_strategy_resize_limit", 2048),

                # Cropping parameters - Can cause blank output if misconfigured
                use_croper=job_input.get("use_croper", False),  # Should be False unless needed
                croper_x=job_input.get("croper_x", 0),
                croper_y=job_input.get("croper_y", 0),
                croper_height=job_input.get("croper_height", 512),
                croper_width=job_input.get("croper_width", 512),

                # Post-processing options - These can interfere with output
                enable_interactive_seg=job_input.get("enable_interactive_seg", False),
                interactive_seg_model=job_input.get("interactive_seg_model", "vit_b"),
                enable_remove_bg=job_input.get("enable_remove_bg", False),
                enable_anime_seg=job_input.get("enable_anime_seg", False),
                enable_realesrgan=job_input.get("enable_realesrgan", False),
                realesrgan_device=job_input.get("realesrgan_device", device.type),
                realesrgan_model=job_input.get("realesrgan_model", "RealESRGAN_x4plus"),
                enable_gfpgan=job_input.get("enable_gfpgan", False),
                gfpgan_device=job_input.get("gfpgan_device", device.type),
                enable_restoreformer=job_input.get("enable_restoreformer", False),
                restoreformer_device=job_input.get("restoreformer_device", device.type)
            )

            # print(f"🔧 Config - sd_keep_unmasked_area: {config.sd_keep_unmasked_area}")
            # print(f"🔧 Config - sd_strength: {config.sd_strength}")
            # print(f"🔧 Config - sd_mask_blur: {config.sd_mask_blur}")
            # print(f"🔧 Config - hd_strategy: {config.hd_strategy}")
            # print(f"🔧 Config - use_croper: {config.use_croper}")

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
