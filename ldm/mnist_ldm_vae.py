import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SpatialVAE(nn.Module):
    """
    Convolutional VAE with a spatial latent of shape (C, 7, 7).
    For MNIST, C=4 gives a 4x7x7 = 196-dim latent, ~52K total parameters.
    """
    def __init__(self, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: 1x28x28 -> 32x14x14 -> 64x7x7 -> 2C x 7x7
        # The final 1x1 conv outputs 2C channels: first C = mu, last C = logvar.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 28 -> 14
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.SiLU(),
            nn.Conv2d(64, 2 * latent_channels, kernel_size=1),      # 7x7, 2C channels
        )

        # Decoder: C x 7x7 -> 64x7x7 -> 32x14x14 -> 1x28x28
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 14 -> 28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)                 # (B, 2C, 7, 7)
        mu, logvar = h.chunk(2, dim=1)      # each (B, C, 7, 7)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl) / x.size(0), recon_loss / x.size(0), kl / x.size(0)


def train_vae(
    epochs=20,
    batch_size=128,
    lr=1e-3,
    latent_channels=4,
    beta=1.0,
    device=None,
    save_path="vae_mnist_spatial.pt",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model = SpatialVAE(latent_channels=latent_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)

        avg = running / len(train_ds)
        print(f"epoch {epoch:02d} | loss {avg:.4f}")

    torch.save({"state_dict": model.state_dict(), "latent_channels": latent_channels}, save_path)
    print(f"saved {save_path}")
    return model


# ---- Helpers for the diffusion stage (spatial-latent versions) ----
@torch.no_grad()
def encode_dataset(model, device=None, batch_size=512):
    """Pre-encode all 60k MNIST images into spatial latents of shape (C, 7, 7)."""
    device = device or next(model.parameters()).device
    model.eval()
    ds = datasets.MNIST(root="./data", train=True, download=True,
                        transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    latents, labels = [], []
    for x, y in loader:
        mu, _ = model.encode(x.to(device))   # deterministic: use mu, not sampled z
        latents.append(mu.cpu())
        labels.append(y)
    return torch.cat(latents), torch.cat(labels)   # (60000, C, 7, 7), (60000,)


@torch.no_grad()
def decode_latents(model, z, device=None):
    """Decode a batch of latents (B, C, 7, 7) back to images (B, 1, 28, 28)."""
    device = device or next(model.parameters()).device
    model.eval()
    return model.decode(z.to(device)).cpu()


if __name__ == "__main__":
    train_vae(epochs=20, latent_channels=4, beta=1.0)

