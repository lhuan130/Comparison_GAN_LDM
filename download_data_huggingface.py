# download_hf_dataset.py
import argparse
import os
import shutil
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download a dataset from Hugging Face using a link and token."
    )
    parser.add_argument(
        "dataset_url",
        type=str,
        help="The full URL of the Hugging Face dataset "
             "(e.g., https://huggingface.co/datasets/zh-plus/tiny-imagenet).",
    )
    parser.add_argument(
        "token",
        type=str,
        help="Your Hugging Face API token.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./downloaded_dataset",
        help="Directory to save the dataset locally (default: ./downloaded_dataset)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory if it already exists before saving.",
    )
    args = parser.parse_args()

    # The datasets library expects a repo_id (e.g., "zh-plus/tiny-imagenet"),
    # not the full URL. Strip the standard prefix if present.
    prefix = "https://huggingface.co/datasets/"
    if args.dataset_url.startswith(prefix):
        repo_id = args.dataset_url[len(prefix):].rstrip("/")
    else:
        repo_id = args.dataset_url.rstrip("/")

    print(f"Attempting to load dataset: {repo_id}")

    try:
        dataset = load_dataset(repo_id, token=args.token)
        print(f"Dataset loaded successfully. Splits available: {list(dataset.keys())}")

        # save_to_disk owns directory creation. Pre-creating the directory
        # conflicts with its contract and can produce nested output or errors.
        if os.path.exists(args.output_dir):
            if args.overwrite:
                print(f"Removing existing directory: {args.output_dir}")
                shutil.rmtree(args.output_dir)
            else:
                print(
                    f"Warning: {args.output_dir} already exists. "
                    f"Pass --overwrite to replace it, or save_to_disk may fail."
                )

        dataset.save_to_disk(args.output_dir)

        # Verify the save actually produced files before claiming success.
        saved_files = os.listdir(args.output_dir)
        if not saved_files:
            raise RuntimeError(
                f"save_to_disk produced no files in {args.output_dir}"
            )
        print(f"Dataset saved locally to: {os.path.abspath(args.output_dir)}")
        print(f"Saved artifacts: {saved_files}")

        # Confirm labels survived the round-trip.
        if "train" in dataset:
            sample = dataset["train"][0]
            print(f"Sample check - Features: {list(sample.keys())}")
            print(f"Sample check - Label: {sample.get('label')}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()

