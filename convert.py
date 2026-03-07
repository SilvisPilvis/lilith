"""
Download PyTorch model and convert to JSON for Burn import.

Usage:
    python download_and_convert.py
    python download_and_convert.py --url https://example.com/model.pt --output weights.json
"""

import json
import os
import urllib.request
from dataclasses import dataclass


EXPECTED_SHAPES = {
    "conv1.weight": [32, 3, 3, 3],
    "conv1.bias": [32],
    "conv2.weight": [64, 32, 3, 3],
    "conv2.bias": [64],
    "conv3.weight": [128, 64, 3, 3],
    "conv3.bias": [128],
    "fc1.weight": [128, 128],
    "fc1.bias": [128],
    "fc2.weight": [1, 128],
    "fc2.bias": [1],
}


@dataclass
class Args:
    url: str = "https://img.vinetaerentraute.id.lv/model.pt"
    output: str = "weights.json"
    keep_pt: bool = False
    downloaded_pt_path: str = "downloaded_model.pt"


def download_model(url: str, output_path: str):
    """Download model from URL."""
    print(f"Downloading model from {url}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0",
    }

    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=300) as response:
        with open(output_path, "wb") as f:
            f.write(response.read())

    print(f"Model saved to {output_path}")


def convert_pt_to_json(pt_path: str, json_path: str):
    """Convert PyTorch .pt file to JSON weights."""
    import torch

    print(f"Loading PyTorch model from {pt_path}...")

    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    print(f"Converting {len(state_dict)} tensors to JSON...")

    weights = {}
    for name, param in state_dict.items():
        values = param.detach().cpu().numpy().tolist()
        weights[name] = values

        expected_shape = EXPECTED_SHAPES.get(name)
        actual_shape = list(param.shape)
        if expected_shape is not None and actual_shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for {name}: expected {expected_shape}, got {actual_shape}"
            )

        print(f"  {name}: {actual_shape}")

    with open(json_path, "w") as f:
        json.dump(weights, f)

    print(f"Weights saved to {json_path}")
    print(f"File size: {os.path.getsize(json_path) / 1024 / 1024:.2f} MB")


def main():
    args = Args()

    pt_path = args.downloaded_pt_path

    download_model(args.url, pt_path)
    convert_pt_to_json(pt_path, args.output)

    if not args.keep_pt:
        os.remove(pt_path)
        print(f"Removed temporary file {pt_path}")


if __name__ == "__main__":
    main()
