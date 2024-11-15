import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# `enable_model_cpu_offload` is for single GPU
pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
lora_path = "/root/data-fs/DimensionX/src/orbit_left_lora_weights.safetensors"
lora_rank = 256
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)
# pipe.to("cuda")


prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
video = pipe(image, prompt, use_dynamic_cfg=True)
export_to_video(video.frames[0], "output.mp4", fps=8)