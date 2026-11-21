# hf_to_studiogan.py
import os
from pathlib import Path
from datasets import load_from_disk
from PIL import Image
import argparse

def convert(hf_path, out_root="data", dataset_name="TINY_ILSVRC2012"):
    ds = load_from_disk(hf_path)
    base = Path(out_root) / dataset_name

    # StudioGAN expects "valid", not "val" or "validation"
    split_map = {"train": "train", "valid": "valid", "validation": "valid"}

    for hf_split, studio_split in split_map.items():
        if hf_split not in ds:
            print(f"Skipping missing split: {hf_split}")
            continue

        split_ds = ds[hf_split]
        print(f"Converting {hf_split} -> {studio_split} ({len(split_ds)} samples)")

        # Use integer labels as folder names; StudioGAN does not require names
        for idx in range(len(split_ds)):
            sample = split_ds[idx]
            label = sample["label"]
            img = sample["image"]

            class_dir = base / studio_split / f"cls{label}"
            class_dir.mkdir(parents=True, exist_ok=True)

            img_path = class_dir / f"{studio_split}{idx}.png"
            img.save(img_path)

    print(f"Done. StudioGAN data root: {base.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_path", help="Path passed to datasets.save_to_disk()")
    ap.add_argument("--out_root", default="data")
    args = ap.parse_args()
    convert(args.hf_path, args.out_root)

