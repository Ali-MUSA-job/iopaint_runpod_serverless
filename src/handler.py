import runpod
import base64
from io import BytesIO
from PIL import Image
from iopaint import iopaint  # Main function from IOPaint repo

# Helper: decode base64 to PIL image
def decode_base64_image(b64_string):
    return Image.open(BytesIO(base64.b64decode(b64_string))).convert("RGB")

# Helper: encode PIL image to base64 PNG
def encode_base64_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def handler(job):
    try:
        job_input = job.get("input", {})
        image_b64 = job_input.get("image")
        mask_b64 = job_input.get("mask")
        model_name = job_input.get("model", "lama")  # default model is 'lama'

        if not image_b64 or not mask_b64:
            return {"error": "Missing 'image' or 'mask' in input."}

        # Decode images
        image = decode_base64_image(image_b64).convert("RGB")
        mask = decode_base64_image(mask_b64).convert("L")  # mask typically single channel

        # Run IOPaint inpainting
        # The iopaint function signature is: iopaint(image, mask, model="lama", ...)
        result_image = iopaint(image, mask, model=model_name)

        # Encode output image to base64
        output_b64 = encode_base64_image(result_image)

        return {"output_image": output_b64}

    except Exception as e:
        return {"error": str(e)}

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
