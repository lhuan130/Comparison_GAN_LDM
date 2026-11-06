import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# ---------------------------------------------------------
# 1. VAE (Variational Autoencoder)
# Role: Compresses pixel images into a low-dimensional latent space
#       and decodes latents back into pixels at the end.
# ---------------------------------------------------------
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    subfolder="vae"
)
# The VAE is frozen during inference; we only use it for encoding/decoding.
vae.requires_grad_(False)
vae.to("cuda")

# ---------------------------------------------------------
# 2. Text Encoder & Tokenizer (CLIP)
# Role: Converts the text prompt into a sequence of embeddings
#       that condition the U-Net.
# ---------------------------------------------------------
tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14"
)
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14"
)
text_encoder.requires_grad_(False)
text_encoder.to("cuda")

# ---------------------------------------------------------
# 3. U-Net (The Denoiser)
# Role: The core neural network. It predicts the noise to remove
#       at each step, conditioned on the text embeddings and timestep.
# ---------------------------------------------------------
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    subfolder="unet"
)
unet.to("cuda")

# ---------------------------------------------------------
# 4. Scheduler
# Role: Defines the mathematical trajectory for adding/removing noise.
#       Here we use PNDMScheduler, but others (e.g., LMS, Euler) can be swapped.
# ---------------------------------------------------------
scheduler = PNDMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    subfolder="scheduler"
)

# ---------------------------------------------------------
# Minimal Manual Inference Loop (Conceptual)
# ---------------------------------------------------------
prompt = ["a photograph of an astronaut riding a horse"]
height, width = 512, 512
num_inference_steps = 50
guidance_scale = 7.5

# a. Tokenize and encode text
text_input = tokenizer(
    prompt, 
    padding="max_length", 
    max_length=tokenizer.model_max_length, 
    truncation=True, 
    return_tensors="pt"
)
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]

# For classifier-free guidance, we also need unconditional embeddings (empty prompt).
# (Simplified here; in practice, you duplicate the batch with empty text.)

# b. Initialize random latent noise (shape: batch, 4 channels, H/8, W/8)
latents = torch.randn(
    (1, 4, height // 8, width // 8), 
    device="cuda"
) * scheduler.init_noise_sigma

# c. Set timesteps
scheduler.set_timesteps(num_inference_steps)

# d. Denoising loop
for t in scheduler.timesteps:
    # Scale model input if required by scheduler
    latent_model_input = scheduler.scale_model_input(latents, t)
    
    # Predict noise with U-Net (conditioned on text)
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=text_embeddings
        ).sample
    
    # Scheduler step: compute previous noisy sample x_t -> x_{t-1}
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# e. Decode the final latents back to pixel space with VAE
latents = 1 / 0.18215 * latents  # Scaling factor specific to SD 1.x
with torch.no_grad():
    image = vae.decode(latents).sample

# Post-process to PIL Image (clamp, scale, convert)
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()
# ... convert numpy array to PIL ...

