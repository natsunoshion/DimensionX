import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from accelerate import Accelerator

from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight

accelerator = Accelerator(device_placement=True, cpu=False)

quantization = int8_weight_only

text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="text_encoder", torch_dtype=torch.bfloat16,
                                            #   device_map="auto"
                                              )
text_encoder.to(accelerator.device)
quantize_(text_encoder, quantization())

transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b-I2V",subfolder="transformer", torch_dtype=torch.bfloat16)
transformer.to(accelerator.device)
quantize_(transformer, quantization())

vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16,
                                            #  device_map="auto"
                                             )
vae.to(accelerator.device)
quantize_(vae, quantization())
# accelerator.print()

print("Loading pipeline...")
pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V",
                                                     text_encoder=text_encoder,
                                                     transformer=transformer,
                                                     vae=vae,
                                                     torch_dtype=torch.float32)
print("Pipeline loaded.")
# lora_path = "/root/data-fs/DimensionX/src/orbit_left_lora_weights.safetensors"
# lora_rank = 256
# pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
# pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)

pipe = accelerator.prepare(pipe)
# text_encoder.to(accelerator.device)
# transformer.to(accelerator.device)
# vae.to(accelerator.device)
print("Accelerator prepared.")

print(f"Text encoder on device: {next(pipe.text_encoder.parameters()).device}")
print(f"Transformer on device: {next(pipe.transformer.parameters()).device}")
print(f"VAE on device: {next(pipe.vae.parameters()).device}")

prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
image = load_image("input.png")

print("Starting inference...")
video = pipe(image, prompt, use_dynamic_cfg=True)
print("Inference completed.")


print("Exporting video...")
export_to_video(video.frames[0], "output.mp4", fps=8)
print("Video export completed.")