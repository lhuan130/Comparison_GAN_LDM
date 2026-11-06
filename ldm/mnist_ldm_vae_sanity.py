# Run after training: verifies the spatial VAE reconstructs digits recognizably.
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from vae_mnist_spatial import SpatialVAE, decode_latents


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load checkpoint and rebuild the model
    ckpt = torch.load("vae_mnist_spatial.pt", map_location=device)
    model = SpatialVAE(latent_channels=ckpt["latent_channels"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()  # important: encode path has no dropout/BN, but be explicit

    # 2. Pull 8 validation images (held-out, never seen during training)
    val = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    x = torch.stack([val[i][0] for i in range(8)]).to(device)  # (8, 1, 28, 28)

    # 3. Encode to spatial latents (deterministic: use mu, not sampled z)
    with torch.no_grad():
        mu, logvar = model.encode(x)          # each (8, C, 7, 7)
        recon = decode_latents(model, mu)     # (8, 1, 28, 28)

    print(f"latent shape: {tuple(mu.shape)}")
    print(f"recon shape:  {tuple(recon.shape)}")

    # 4. Save originals vs reconstructions side by side
    #    Top row = originals, bottom row = reconstructions
    grid = make_grid(torch.cat([x.cpu(), recon]), nrow=8, padding=2)
    save_image(grid, "vae_recon_spatial.png")
    print("saved vae_recon_spatial.png")

    # 5. Optional: visualize the 4 latent channels as heatmaps
    #    Normalize per-channel to [0,1] so structure is visible regardless of scale.
    lat = mu[0].cpu()                                  # (C, 7, 7) for the first image
    lat_min = lat.amin(dim=(1, 2), keepdim=True)
    lat_max = lat.amax(dim=(1, 2), keepdim=True)
    lat_norm = (lat - lat_min) / (lat_max - lat_min + 1e-8)
    # Upsample 7x7 -> 28x28 so the heatmaps are the same size as the digits
    lat_up = torch.nn.functional.interpolate(
        lat_norm.unsqueeze(0), size=(28, 28), mode="nearest"
    )                                                  # (1, C, 28, 28)
    save_image(lat_up, "vae_latent_channels.png", nrow=lat.shape[0])
    print("saved vae_latent_channels.png")


if __name__ == "__main__":
    main()
