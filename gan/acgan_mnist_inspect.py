import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from acgan import Generator, Discriminator, LATENT_DIM, NUM_CLASSES, DEVICE
# ^ adjust import to wherever your model classes live; if you kept everything
#   in one file, just import from that file instead.

# ----------------------------------------------------------------------
# 1. Load a checkpoint into fresh models (eval mode)
# ----------------------------------------------------------------------
def load_checkpoint(path, device=DEVICE):
    G = Generator(LATENT_DIM, NUM_CLASSES).to(device)
    D = Discriminator(NUM_CLASSES).to(device)
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    G.eval()
    D.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded {path} (epoch {epoch}) on {device}")
    return G, D

# ----------------------------------------------------------------------
# 2. Fixed-noise grid across all labels
#    Useful for visually checking conditioning: does the same z produce
#    a recognizably different digit for each label?
# ----------------------------------------------------------------------
@torch.no_grad()
def label_grid(G, n_per_class=8, seed=0, path="inspect_label_grid.png"):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    z = torch.randn(n_per_class, LATENT_DIM, device=DEVICE, generator=g)
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class,
                             figsize=(n_per_class, NUM_CLASSES))
    for label in range(NUM_CLASSES):
        labels = torch.full((n_per_class,), label, dtype=torch.long, device=DEVICE)
        imgs = G(z, labels).cpu()
        for j in range(n_per_class):
            ax = axes[label, j]
            ax.imshow(imgs[j, 0], cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(str(label), rotation=0, labelpad=20, va="center")
    plt.suptitle(f"Same noise vector, all labels (seed={seed})")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

# ----------------------------------------------------------------------
# 3. Same label, varying noise
#    Shows intra-class diversity: is the generator producing variety,
#    or has it collapsed to one prototype per class?
# ----------------------------------------------------------------------
@torch.no_grad()
def diversity_grid(G, label=3, n=16, seed=0, path=None):
    path = path or f"inspect_div_label{label}.png"
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    z = torch.randn(n, LATENT_DIM, device=DEVICE, generator=g)
    labels = torch.full((n,), label, dtype=torch.long, device=DEVICE)
    imgs = G(z, labels).cpu()
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = axes.flatten()
    for j in range(n):
        axes[j].imshow(imgs[j, 0], cmap="gray", vmin=-1, vmax=1)
        axes[j].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"Label {label}: {n} samples (seed={seed})")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

# ----------------------------------------------------------------------
# 4. Quantitative conditioning check
#    Generate N samples per label and ask D's own classifier head
#    how many it would assign to the requested label. High diagonal
#    = conditioning works. Off-diagonal mass tells you *which*
#    classes are being confused.
# ----------------------------------------------------------------------
@torch.no_grad()
def conditioning_matrix(G, D, n_per_class=200, path="inspect_confusion.png"):
    D.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for label in range(NUM_CLASSES):
        z = torch.randn(n_per_class, LATENT_DIM, device=DEVICE)
        labels = torch.full((n_per_class,), label, dtype=torch.long, device=DEVICE)
        imgs = G(z, labels)
        _, cls_logits = D(imgs)
        preds = cls_logits.argmax(1).cpu()
        for p in preds:
            conf[label, p.item()] += 1
    acc = conf.diagonal().float().mean().item()
    print(f"D-classifier agreement with requested labels: {acc:.3f}")
    print("Rows = requested label, columns = D's prediction")

    # normalize rows for display
    conf_norm = conf.float() / conf.sum(1, keepdim=True).clamp(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("D predicted label")
    ax.set_ylabel("G requested label")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = conf_norm[i, j].item()
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")
    return conf_norm

# ----------------------------------------------------------------------
# 5. Interpolation in latent space for a fixed label
#    A smooth row of digits suggests a well-behaved latent manifold.
#    Jumps/flickers are a sign of a poorly organized latent space
#    (common when the classifier head dominated training).
# ----------------------------------------------------------------------
@torch.no_grad()
def latent_interp(G, label=3, n_steps=10, seed=0, path=None):
    path = path or f"inspect_interp_label{label}.png"
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    z0 = torch.randn(1, LATENT_DIM, device=DEVICE, generator=g)
    z1 = torch.randn(1, LATENT_DIM, device=DEVICE, generator=g)
    alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(-1, 1, 1)
    zs = ((1 - alphas) * z0.unsqueeze(0) + alphas * z1.unsqueeze(0)).squeeze(1)
    labels = torch.full((n_steps,), label, dtype=torch.long, device=DEVICE)
    imgs = G(zs, labels).cpu()
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps, 1.3))
    for j in range(n_steps):
        axes[j].imshow(imgs[j, 0], cmap="gray", vmin=-1, vmax=1)
        axes[j].axis("off")
        axes[j].set_title(f"{j/(n_steps-1):.1f}", fontsize=7)
    plt.suptitle(f"Latent interpolation, label {label}")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

# ----------------------------------------------------------------------
# 6. Quick check on real MNIST: how well does D's classifier head
#    do on actual digits? If this is bad, the generator's training
#    signal was bad regardless of what the samples look like.
# ----------------------------------------------------------------------
@torch.no_grad()
def real_classifier_accuracy(D, n_batches=20, batch_size=256):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5])])
    ds = datasets.MNIST(root="./data", train=False, download=True, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    correct = total = 0
    for i, (imgs, labels) in enumerate(dl):
        if i >= n_batches:
            break
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        _, cls_logits = D(imgs)
        correct += (cls_logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"D classifier accuracy on real MNIST: {correct/total:.3f}")

# ----------------------------------------------------------------------
# Main: run all inspection checks against one checkpoint
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/acgan_epoch50.pt"
    G, D = load_checkpoint(ckpt_path)

    label_grid(G, n_per_class=8, seed=0)
    diversity_grid(G, label=3, n=16, seed=1)
    conditioning_matrix(G, D, n_per_class=200)
    latent_interp(G, label=3, n_steps=10, seed=2)
    real_classifier_accuracy(D)

