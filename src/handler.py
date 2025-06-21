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
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)
    if image_np.shape[-1] == 3:
        image_np = image_np[..., ::-1]
    image = Image.fromarray(image_np)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def normalize_mask(mask_np):
    print(f"🔍 Original mask shape: {mask_np.shape}")
    mask_np = np.squeeze(mask_np)
    print(f"🔍 After squeeze: {mask_np.shape}")
    return mask_np[:, :, 0] if mask_np.ndim == 3 else mask_np


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

                # Stable Diffusion parameters
                sd_steps=job_input.get("sd_steps", 50),
                sd_guidance_scale=job_input.get("sd_guidance_scale", 7.5),
                sd_seed=job_input.get("sd_seed", 42),
                sd_strength=job_input.get("sd_strength", 1.0),
                sd_keep_unmasked_area=job_input.get("sd_keep_unmasked_area", True),
                sd_num_samples=job_input.get("sd_num_samples", 1),
                sd_match_histograms=job_input.get("sd_match_histograms", False),
                sd_mask_blur=job_input.get("sd_mask_blur", 11),
                sd_freeu=job_input.get("sd_freeu", False),
                sd_freeu_config=job_input.get("sd_freeu_config", {}),
                sd_lcm_lora=job_input.get("sd_lcm_lora", False),

                # HD Strategy
                hd_strategy=job_input.get("hd_strategy", "Original"),
                hd_strategy_crop_margin=job_input.get("hd_strategy_crop_margin", 32),
                hd_strategy_crop_trigger_size=job_input.get("hd_strategy_crop_trigger_size", 512),
                hd_strategy_resize_limit=job_input.get("hd_strategy_resize_limit", 2048),

                # Cropping parameters
                use_croper=job_input.get("use_croper", False),
                croper_x=job_input.get("croper_x", 0),
                croper_y=job_input.get("croper_y", 0),
                croper_height=job_input.get("croper_height", 512),
                croper_width=job_input.get("croper_width", 512),

                # Post-processing options
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
            result = current_model(image_np, mask_np, config)

        result_b64 = encode_base64_image(result)
        return {"output_image": result_b64}

    except Exception as e:
        print(f"❌ Handler error: {e}")
        return {"error": str(e)}


# RunPod Serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
