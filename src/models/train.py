from pathlib import Path
from typing import Any, Dict

import mlflow
from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    vision_learner,
    resnet18,
    resnet34,
    accuracy,
)

from src.data.download_data import main as download_data_main
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_arch(arch_name: str):
    """Map a string in config to a fastai architecture."""
    arch_name = arch_name.lower()
    if arch_name == "resnet18":
        return resnet18
    elif arch_name == "resnet34":
        return resnet34
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")


def create_dataloaders(
    images_root: Path,
    cfg: Dict[str, Any],
):
    """Create fastai ImageDataLoaders from the images folder."""
    image_size = cfg["image_size"]
    batch_size = cfg["batch_size"]
    valid_pct = cfg["valid_pct"]
    random_seed = cfg["random_seed"]

    set_seed(random_seed)
    # Proper formatting for use of ImageDataLoaders with config file generated with help from ChatGPT.
    dls = ImageDataLoaders.from_folder(
        images_root,
        valid_pct=valid_pct,
        seed=random_seed,
        item_tfms=Resize(image_size),
        bs=batch_size,
    )
    return dls


def train() -> None:
    """Main training function."""
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    # Ensure data is present/downloads from GCS if needed)
    download_data_main()

    images_root = Path(data_cfg["extract_dir"]) / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found at {images_root.resolve()}")

    print(f"[train] Using images from: {images_root}")

    # Build DataLoaders
    dls = create_dataloaders(images_root, train_cfg)

    # Build model
    arch = get_arch(model_cfg["arch"])
    learn = vision_learner(dls, arch, metrics=accuracy)

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    model_dir = Path(train_cfg["model_dir"]).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    export_path = model_dir / train_cfg["export_name"]

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "arch": model_cfg["arch"],
                "image_size": train_cfg["image_size"],
                "batch_size": train_cfg["batch_size"],
                "valid_pct": train_cfg["valid_pct"],
                "random_seed": train_cfg["random_seed"],
                "epochs": train_cfg["epochs"],
            }
        )

        print(f"[train] Starting training for {train_cfg['epochs']} epochs...")
        learn.fine_tune(train_cfg["epochs"])

        val_loss, val_acc = learn.validate()
        val_loss = float(val_loss)
        val_acc = float(val_acc)

        print(f"[train] Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

        # Logging with mlflow debugged with help of ChatGPT
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_acc)

        # Export model artifact (absolute path)
        learn.export(export_path)
        print(f"[train] Exported model to {export_path}")

        mlflow.log_artifact(str(export_path), artifact_path="model")


    print("[train] Training run complete.")


if __name__ == "__main__":
    train()
