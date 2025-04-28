```markdown
# SHAP-E Quickstart

This repository provides a simple guide to generate 3D assets from text prompts using [OpenAI's SHAP-E](https://github.com/openai/shap-e).

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/openai/shap-e
cd shap-e
pip install -e .
```

## Quickstart

### 1. Import Libraries

```python
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh
```

### 2. Setup Device

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3. Load Models

```python
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
```

### 4. Generate Latents from a Prompt

```python
batch_size = 1
guidance_scale = 15.0
prompt = "a donut"

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
```

### 5. Render Latents into GIFs

```python
render_mode = 'nerf'  # Alternatives: 'stf'
size = 64  # Render size (higher = slower but better quality)

cameras = create_pan_cameras(size, device)

for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    display(gif_widget(images))
```

### 6. Export 3D Mesh Files

```python
for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'example_mesh_{i}.ply', 'wb') as f:
        t.write_ply(f)
    with open(f'example_mesh_{i}.obj', 'w') as f:
        t.write_obj(f)
```

## Outputs

- Animated GIFs displaying a rotating 3D view of the generated object.
- 3D model files in `.ply` and `.obj` formats, ready for editing in 3D software (like Blender).

## Notes

- A CUDA-enabled GPU is highly recommended for faster generation.
- You can tweak `guidance_scale` to control how closely outputs match the prompt.
- Use the `.obj` files for further customization in your favorite 3D design tools.

## License

This project follows [OpenAI's shap-e license](https://github.com/openai/shap-e/blob/main/LICENSE).
```
