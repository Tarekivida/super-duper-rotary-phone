SHAP-E Quickstart
This guide shows you how to quickly get started with OpenAI's SHAP-E to generate 3D assets from text prompts.

Installation
Clone the repository and install the package in editable mode:

bash
Copy
Edit
git clone https://github.com/openai/shap-e
cd shap-e
pip install -e .
Usage
First, import the necessary libraries and configure your device:

python
Copy
Edit
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Load the models and diffusion configuration:

python
Copy
Edit
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
Generate Latents from a Text Prompt
Set your generation parameters:

python
Copy
Edit
batch_size = 1
guidance_scale = 15.0
prompt = "a donut"
Sample latents based on the prompt:

python
Copy
Edit
latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=4,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)
Render Images as GIFs
Render the generated latents into a panoramic GIF:

python
Copy
Edit
render_mode = 'nerf'  # Alternatives: 'stf'
size = 64             # Render size; higher values increase quality and computation time

cameras = create_pan_cameras(size, device)

for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    display(gif_widget(images))
Export 3D Meshes
You can also decode and save 3D meshes in .ply and .obj formats for later use in 3D tools like Blender:

python
Copy
Edit
for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'example_mesh_{i}.ply', 'wb') as f:
        t.write_ply(f)
    with open(f'example_mesh_{i}.obj', 'w') as f:
        t.write_obj(f)
Output
Panoramic GIF preview of the generated object

3D mesh files (.ply and .obj) ready for editing in software like Blender

Notes
A CUDA-enabled GPU is strongly recommended for faster inference.

Adjust guidance_scale for stronger or weaker conditioning on the prompt.

Rendering size and quality can be tuned through the size parameter.

