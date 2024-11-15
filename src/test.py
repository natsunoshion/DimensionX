# To get started, PytorchAO needs to be installed from the GitHub source and PyTorch Nightly.
# Source and nightly installation is only required until the next release.

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only

quantization = int8_weight_only

text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="text_encoder", torch_dtype=torch.float16)
quantize_(text_encoder, quantization())

transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b-I2V",subfolder="transformer", torch_dtype=torch.float16)
quantize_(transformer, quantization())

vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.float16)
quantize_(vae, quantization())

# Create pipeline and run inference
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)