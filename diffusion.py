import torch
from diffusers import StableDiffusionPipeline

model_id = 'OFA-Sys/small-stable-diffusion-v0'
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)
pipe = pipe.to('cuda')
prompt = f'A man who is rock and happy.'
#prompt = f'A man who is {pred_emotion} and {pred_style}.'
image = pipe(prompt).images[0]
image.save('test.png')

