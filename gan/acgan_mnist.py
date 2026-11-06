import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# ----------------------------------------------------------------------
# 1. Hyperparameters
# ----------------------------------------------------------------------
BATCH_SIZE = 128
LATENT_DIM = 100
NUM_CLASSES = 10
IMG_SIZE = 28
CHANNELS = 1
NUM_EPOCHS = 50
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs", exist_ok=True)

# ----------------------------------------------------------------------
# 2. Data loading
# ----------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
)

# ----------------------------------------------------------------------
# 3. Generator
#    Input: (noise z, label y) → concatenated → 28x28 image in [-1,1]
# ----------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.init_size = IMG_SIZE // 4  # 7
        self.fc = nn.Linear(latent_dim * 2, 128 * self.init_size * self.init_size)

        self.net = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),                 # 14x14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),                 # 28x28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, CHANNELS, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # z: (B, latent_dim), labels: (B,)
        label_vec = self.label_emb(labels)               # (B, latent_dim)
        x = torch.cat([z, label_vec], dim=1)             # (B, 2*latent_dim)
        x = self.fc(x)
        x = x.view(x.size(0), 128, self.init_size, self.init_size)
        return self.net(x)

# ----------------------------------------------------------------------
# 4. Discriminator with two heads: validity + auxiliary classifier
# ----------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def block(in_c, out_c, stride=2, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *block(CHANNELS, 64, stride=2, bn=False),    # 14x14
            *block(64, 128, stride=2),                   # 7x7
            *block(128, 256, stride=2),                  # 4x4 (ceil)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)              # (B, 256, 1, 1)

        # Validity head: real/fake
        self.validity = nn.Sequential(
            nn.Linear(256, 1),
            # no sigmoid here — use BCEWithLogitsLoss
        )
        # Auxiliary classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
            # no softmax here — use CrossEntropyLoss
        )

    def forward(self, x):
        feats = self.features(x)
        feats = self.pool(feats).view(x.size(0), -1)     # (B, 256)
        validity = self.validity(feats)                  # (B, 1) logits
        class_logits = self.classifier(feats)            # (B, 10) logits
        return validity, class_logits

# ----------------------------------------------------------------------
# 5. Instantiate models, losses, optimizers
# ----------------------------------------------------------------------
G = Generator(LATENT_DIM, NUM_CLASSES).to(DEVICE)
D = Discriminator(NUM_CLASSES).to(DEVICE)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

G.apply(weights_init)
D.apply(weights_init)

criterion_validity = nn.BCEWithLogitsLoss()
criterion_class = nn.CrossEntropyLoss()

opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

# ----------------------------------------------------------------------
# 6. Training loop
# ----------------------------------------------------------------------
def train():
    for epoch in range(1, NUM_EPOCHS + 1):
        d_loss_total = 0.0
        g_loss_total = 0.0
        d_val_acc_correct = 0
        d_cls_correct = 0
        seen = 0

        for i, (real_imgs, real_labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            B = real_imgs.size(0)

            # -----------------------------------------------------
            # Train Discriminator
            # -----------------------------------------------------
            opt_D.zero_grad()

            # Real batch
            real_valid, real_cls = D(real_imgs)
            real_targets = torch.ones(B, 1, device=DEVICE)
            loss_D_real_valid = criterion_validity(real_valid, real_targets)
            loss_D_real_cls = criterion_class(real_cls, real_labels)

            # Fake batch
            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            fake_labels = torch.randint(0, NUM_CLASSES, (B,), device=DEVICE)
            fake_imgs = G(z, fake_labels).detach()

            fake_valid, _ = D(fake_imgs)
            fake_targets = torch.zeros(B, 1, device=DEVICE)
            loss_D_fake_valid = criterion_validity(fake_valid, fake_targets)
            # NOTE: do NOT compute classification loss on fake images for D.
            # (Misclassifying fakes is the generator's fault.)

            loss_D = loss_D_real_valid + loss_D_fake_valid + loss_D_real_cls
            loss_D.backward()
            opt_D.step()

            # -----------------------------------------------------
            # Train Generator
            # -----------------------------------------------------
            opt_G.zero_grad()

            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            gen_labels = torch.randint(0, NUM_CLASSES, (B,), device=DEVICE)
            gen_imgs = G(z, gen_labels)

            gen_valid, gen_cls = D(gen_imgs)

            # Fool D: want validity ≈ 1
            loss_G_valid = criterion_validity(gen_valid, torch.ones(B, 1, device=DEVICE))
            # Match intended label
            loss_G_cls = criterion_class(gen_cls, gen_labels)

            loss_G = loss_G_valid + loss_G_cls
            loss_G.backward()
            opt_G.step()

            # -----------------------------------------------------
            # Stats
            # -----------------------------------------------------
            d_loss_total += loss_D.item()
            g_loss_total += loss_G.item()

            with torch.no_grad():
                d_val_acc_correct += (
                    (torch.sigmoid(real_valid) > 0.5).sum().item()
                    + (torch.sigmoid(fake_valid) < 0.5).sum().item()
                )
                d_cls_correct += (real_cls.argmax(1) == real_labels).sum().item()
                seen += 2 * B

        avg_d = d_loss_total / len(train_loader)
        avg_g = g_loss_total / len(train_loader)
        val_acc = d_val_acc_correct / seen
        cls_acc = d_cls_correct / (seen / 2)

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"D loss {avg_d:.4f} | G loss {avg_g:.4f} | "
            f"D validity acc {val_acc:.3f} | D class acc {cls_acc:.3f}"
        )

        # Save a checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(
                {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                    "epoch": epoch,
                },
                f"outputs/acgan_epoch{epoch}.pt",
            )

# ----------------------------------------------------------------------
# 7. Visualize: same noise vector, all 10 labels
# ----------------------------------------------------------------------
@torch.no_grad()
def sample_grid(path="outputs/samples.png", n_per_class=8):
    import matplotlib.pyplot as plt

    G.eval()
    fixed_z = torch.randn(n_per_class, LATENT_DIM, device=DEVICE)
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class, figsize=(n_per_class, NUM_CLASSES))
    for label in range(NUM_CLASSES):
        labels = torch.full((n_per_class,), label, dtype=torch.long, device=DEVICE)
        imgs = G(fixed_z, labels).cpu()
        for j in range(n_per_class):
            ax = axes[label, j]
            ax.imshow(imgs[j, 0], cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(str(label), rotation=0, labelpad=20, va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    G.train()

if __name__ == "__main__":
    train()
    sample_grid()

