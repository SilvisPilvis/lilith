"""
PyTorch training script for Kaggle with Gelbooru image fetching.
Images are fetched from Gelbooru CDN, labels are loaded from a separate JSON file.

Usage on Kaggle:
    python train_pytorch_api.py --data_dir . --output_dir /kaggle/working/

Labels JSON format (train_labels.json, valid_labels.json):
{
    "abcdef123456.jpg": 0.85,
    "ghijk789012.jpg": 0.23,
    ...
}

Image URLs are constructed as: https://img2.gelbooru.com/images/ab/cd/abcdef123456.jpg
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional
from IPython.display import display

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

GELBOORU_API_KEY = "09098bd3c6d4cb8e5d9fdea975ddfcc3ccf6042c87ed1936bbf1487181881e3d"
GELBOORU_USER_ID = "1643096"
GELBOORU_API_URL = "https://gelbooru.com/index.php?page=dapi&s=post&q=index"

warnings.simplefilter("ignore", Image.DecompressionBombWarning)


@dataclass
class Args:
    num_epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4

    data_dir: str = "/kaggle/input/datasets/silvestrslignickis/img-regression-labels"
    posts_path: Optional[str] = None
    output_dir: str = "/kaggle/working/output"
    cache_dir: str = "/kaggle/working/image_cache"
    save_best_by: str = "acc_010"
    accuracy_tolerances: tuple[float, ...] = (0.05, 0.1, 0.2)


class ImagePreferenceModel(nn.Module):
    """PyTorch model matching the Burn ImagePreferenceModel architecture."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.nn.functional.silu(self.conv1(x))
        x = torch.nn.functional.silu(self.conv2(x))
        x = torch.nn.functional.silu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.silu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def fetch_posts(page_size: int, page: int) -> list | None:
    """Fetch image posts from Gelbooru API."""
    url = (
        f"{GELBOORU_API_URL}"
        f"&limit={page_size}"
        f"&pid={page}"
        f"&api_key={GELBOORU_API_KEY}"
        f"&user_id={GELBOORU_USER_ID}"
        f"&json=1"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0",
        "Referer": "https://gelbooru.com/",
        "Accept": "application/json",
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            posts = []
            for post in data.get("post", []):
                if "file_url" in post and "image" in post:
                    posts.append(
                        {
                            "id": post.get("id"),
                            "file_url": post["file_url"],
                            "filename": post["image"],
                        }
                    )
            return posts
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"Error fetching posts: {e}")
        return None


def fetch_image(url: str) -> bytes | None:
    """Fetch image data with proper headers (curl-like)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0",
        "Referer": "https://gelbooru.com/",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                print("Got HTML instead of image")
                return None
            return response.read()
    except urllib.error.URLError as e:
        print(f"Error fetching image: {e}")
        return None


def preprocess_with_letterbox(
    img: Image.Image, target_size: tuple = (224, 224)
) -> torch.Tensor:
    """Preprocess image with letterboxing (matching Burn preprocessing)."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    elif img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")

    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (128, 128, 128, 255))
        img = Image.alpha_composite(background, img).convert("RGB")

    target_w, target_h = target_size
    orig_w, orig_h = img.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas.paste(img_resized, (x_offset, y_offset))

    img_array = np.array(canvas).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

    return img_tensor


def build_gelbooru_image_url(filename: str) -> str:
    """Build Gelbooru image URL from filename.

    Format: https://img2.gelbooru.com/images/{dir1}/{dir2}/{filename}
    Example: abcdef.jpg -> https://img2.gelbooru.com/images/ab/cd/abcdef.jpg
    """
    if len(filename) < 4:
        raise ValueError(f"Filename too short: {filename}")

    dir1 = filename[:2]
    dir2 = filename[2:4]

    return f"https://img2.gelbooru.com/images/{dir1}/{dir2}/{filename}"


class GelbooruDataset(Dataset):
    """Dataset that fetches images from Gelbooru."""

    def __init__(self, labels_path: str, cache_dir: str = "./image_cache"):
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        self.filenames = list(self.labels.keys())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loaded {len(self.filenames)} images from {labels_path}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        preference = float(self.labels[filename])

        cache_path = self.cache_dir / filename

        if cache_path.exists():
            image = Image.open(cache_path)
        else:
            try:
                # image_url = build_gelbooru_image_url(filename)
                encoded_filename = urllib.parse.quote(filename, safe="")
                image_url = f"https://img.vinetaerentraute.id.lv/{encoded_filename}"
            except ValueError as e:
                print(f"Invalid filename {filename}: {e}")
                image = Image.new("RGB", (224, 224), (128, 128, 128))
                image = preprocess_with_letterbox(image)
                return image, preference

            image_data = fetch_image(image_url)

            if image_data is None:
                print(f"Failed to fetch {filename}, using placeholder")
                image = Image.new("RGB", (224, 224), (128, 128, 128))
            else:
                try:
                    image = Image.open(BytesIO(image_data))
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(cache_path)
                except Exception as e:
                    print(f"Failed to decode {filename}: {e}")
                    image = Image.new("RGB", (224, 224), (128, 128, 128))

        image = preprocess_with_letterbox(image)
        return image, preference


class LocalDataset(Dataset):
    """Dataset for images with labels JSON (filename -> file_url mapping)."""

    def __init__(
        self, labels_path: str, posts_path: str, cache_dir: str = "./image_cache"
    ):
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        with open(posts_path, "r") as f:
            self.posts = json.load(f)

        self.filenames = [fn for fn in self.labels.keys() if fn in self.posts]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loaded {len(self.filenames)} images with labels and URLs")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        preference = float(self.labels[filename])
        file_url = self.posts[filename]

        cache_path = self.cache_dir / filename

        if cache_path.exists():
            image = Image.open(cache_path)
        else:
            image_data = fetch_image(file_url)

            if image_data is None:
                print(f"Failed to fetch {filename}, using placeholder")
                image = Image.new("RGB", (224, 224), (128, 128, 128))
            else:
                try:
                    image = Image.open(BytesIO(image_data))
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(cache_path)
                except Exception as e:
                    print(f"Failed to decode {filename}: {e}")
                    image = Image.new("RGB", (224, 224), (128, 128, 128))

        image = preprocess_with_letterbox(image)
        return image, preference

    def clear_cache(self):
        os.remove(self.cache_dir)


def export_to_onnx(model, output_path, device):
    """Export model to ONNX format for Burn import."""
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"ONNX model exported to {output_path}")
    except Exception as e:
        print(f"Failed to export ONNX: {e}")


def export_weights_to_json(model, output_path):
    """Export model weights to JSON format for Burn import."""
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()

    with open(output_path, "w") as f:
        json.dump(weights, f)

    print(f"Weights exported to {output_path}")


def mean_absolute_error(preds, targets):
    preds_arr = np.asarray(preds, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)
    return float(np.mean(np.abs(preds_arr - targets_arr)))


def spearman_correlation(preds, targets):
    if len(preds) < 2:
        return 0.0
    corr = spearmanr(preds, targets).statistic
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def tolerance_accuracy(preds, targets, tolerance):
    preds_arr = np.asarray(preds, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)
    errors = np.abs(preds_arr - targets_arr)
    return float(np.mean(errors <= tolerance))


def bucket_accuracy(preds, targets, num_buckets=10):
    preds_arr = np.asarray(preds, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)

    pred_bins = np.clip((preds_arr * num_buckets).astype(np.int32), 0, num_buckets - 1)
    target_bins = np.clip(
        (targets_arr * num_buckets).astype(np.int32), 0, num_buckets - 1
    )

    return float(np.mean(pred_bins == target_bins))


def export_weights_to_safetensors(model, output_path):
    """Export model weights to safetensors format."""
    try:
        from safetensors.torch import save_file

        state_dict = {name: param for name, param in model.named_parameters()}
        save_file(state_dict, output_path)
        print(f"Weights exported to {output_path}")
    except ImportError:
        print("safetensors not installed, skipping...")


def export_weights_to_pytorch(model, output_path):
    """Export full PyTorch state dict."""
    torch.save(model.state_dict(), output_path)
    print(f"PyTorch weights saved to {output_path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ImagePreferenceModel().to(device)

    existing_model_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(existing_model_path):
        print(f"Loading existing model from {existing_model_path}")
        model.load_state_dict(torch.load(existing_model_path, map_location=device, weights_only=True))
        print("Resuming training from existing model")
    else:
        print("No existing model found, starting from scratch")

    if args.posts_path:
        train_dataset = LocalDataset(
            os.path.join(args.data_dir, "train_labels.json"),
            os.path.join(args.data_dir, args.posts_path),
            args.cache_dir,
        )
        valid_dataset = LocalDataset(
            os.path.join(args.data_dir, "valid_labels.json"),
            os.path.join(args.data_dir, args.posts_path),
            args.cache_dir,
        )
    else:
        train_dataset = GelbooruDataset(
            os.path.join(args.data_dir, "train_labels.json"),
            args.cache_dir,
        )
        valid_dataset = GelbooruDataset(
            os.path.join(args.data_dir, "valid_labels.json"),
            args.cache_dir,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_val_spearman = float("-inf")
    best_val_acc_010 = 0.0

    epoch_history = []

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0

        for images, preferences in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"
        ):
            images = images.to(device).float()
            preferences = preferences.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, preferences)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, preferences in valid_loader:
                images = images.to(device).float()
                preferences = preferences.to(device).unsqueeze(1).float()
                outputs = model(images)
                loss = criterion(outputs, preferences)
                val_loss += loss.item()

                all_preds.extend(outputs.squeeze(1).cpu().numpy().tolist())
                all_targets.extend(preferences.squeeze(1).cpu().numpy().tolist())

        val_loss /= len(valid_loader)
        val_mae = mean_absolute_error(all_preds, all_targets)
        val_spearman = spearman_correlation(all_preds, all_targets)
        tolerance_metrics = {
            tol: tolerance_accuracy(all_preds, all_targets, tol)
            for tol in args.accuracy_tolerances
        }
        val_bucket_acc = bucket_accuracy(all_preds, all_targets)

        tolerance_summary = ", ".join(
            f"Acc@{tol:.2f}={acc:.4f}" for tol, acc in tolerance_metrics.items()
        )

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Val MAE={val_mae:.4f}, Val Spearman={val_spearman:.4f}, "
            f"BucketAcc={val_bucket_acc:.4f}, {tolerance_summary}"
        )

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_spearman": val_spearman,
            "bucket_acc": val_bucket_acc,
            "acc_005": tolerance_metrics[0.05],
            "acc_010": tolerance_metrics[0.1],
            "acc_020": tolerance_metrics[0.2],
        })

        should_save = False
        if args.save_best_by == "loss":
            should_save = val_loss < best_val_loss
        elif args.save_best_by == "mae":
            should_save = val_mae < best_val_mae
        elif args.save_best_by == "acc_010":
            should_save = tolerance_metrics[0.1] > best_val_acc_010
        else:
            should_save = val_spearman > best_val_spearman

        if should_save:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_val_spearman = val_spearman
            best_val_acc_010 = tolerance_metrics[0.1]
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pt")
            )
            print(
                "  -> Saved best model "
                f"(loss={val_loss:.4f}, mae={val_mae:.4f}, spearman={val_spearman:.4f}, "
                f"acc_010={tolerance_metrics[0.1]:.4f})"
            )

    print("\nTraining complete!")

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - All Epochs")
    print("=" * 80)

    df = pl.DataFrame(epoch_history)
    df = df.with_columns([
        pl.col("train_loss").round(4),
        pl.col("val_loss").round(4),
        pl.col("val_mae").round(4),
        pl.col("val_spearman").round(4),
        pl.col("bucket_acc").round(4),
        pl.col("acc_005").round(4),
        pl.col("acc_010").round(4),
        pl.col("acc_020").round(4),
    ])

    best_loss_idx = df["val_loss"].arg_min()
    best_mae_idx = df["val_mae"].arg_min()
    best_spearman_idx = df["val_spearman"].arg_max()
    best_acc005_idx = df["acc_005"].arg_max()
    best_acc010_idx = df["acc_010"].arg_max()
    best_acc020_idx = df["acc_020"].arg_max()
    best_bucket_idx = df["bucket_acc"].arg_max()

    print("\nBest values achieved:")
    print(f"  Best Val Loss:    {df['val_loss'].min():.4f} (epoch {df['epoch'][best_loss_idx]})")
    print(f"  Best Val MAE:     {df['val_mae'].min():.4f} (epoch {df['epoch'][best_mae_idx]})")
    print(f"  Best Spearman:    {df['val_spearman'].max():.4f} (epoch {df['epoch'][best_spearman_idx]})")
    print(f"  Best Acc@0.05:    {df['acc_005'].max():.4f} (epoch {df['epoch'][best_acc005_idx]})")
    print(f"  Best Acc@0.10:    {df['acc_010'].max():.4f} (epoch {df['epoch'][best_acc010_idx]})")
    print(f"  Best Acc@0.20:    {df['acc_020'].max():.4f} (epoch {df['epoch'][best_acc020_idx]})")
    print(f"  Best BucketAcc:   {df['bucket_acc'].max():.4f} (epoch {df['epoch'][best_bucket_idx]})")

    # print("\nFull epoch history (sortable table):")
    # print(df)
    

    csv_path = os.path.join(args.output_dir, "training_history.csv")
    df.write_csv(csv_path)
    print(f"\nTraining history saved to {csv_path}")

    # if args.posts_path:
    #     train_dataset.clear_cache()

    # os.remove('/kaggle/working/image_cache')

    export_weights_to_json(model, os.path.join(args.output_dir, "weights.json"))
    export_weights_to_safetensors(
        model, os.path.join(args.output_dir, "model.safetensors")
    )
    export_weights_to_pytorch(model, os.path.join(args.output_dir, "model.pt"))
    export_to_onnx(model, os.path.join(args.output_dir, "model.onnx"), device)

    print(f"\nAll files saved to {args.output_dir}")

    display(df)


def main():
    args = Args()

    os.makedirs(args.output_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
