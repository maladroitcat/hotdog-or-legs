from pathlib import Path
from io import BytesIO
from typing import Dict, Any

import requests
from fastai.learner import load_learner
from fastai.vision.all import PILImage

from src.utils.config import load_config


_model = None  # cached model


def get_model():
    """Load and cache the trained fastai Learner."""
    global _model
    if _model is None:
        config = load_config()
        train_cfg = config["training"]

        model_dir = Path(train_cfg["model_dir"])
        export_name = train_cfg["export_name"]
        export_path = (model_dir / export_name).resolve()

        if not export_path.exists():
            raise FileNotFoundError(f"Model file not found at {export_path}")

        print(f"[predict] Loading model from {export_path}")
        _model = load_learner(export_path)
        print("[predict] Model loaded.")
    return _model


def predict_from_url(image_url: str) -> Dict[str, Any]:
    """
    Download an image from a URL, run the classifier, and return predictions.
    """
    resp = requests.get(image_url)
    resp.raise_for_status()

    img = PILImage.create(BytesIO(resp.content))

    learn = get_model()
    pred_class, pred_idx, probs = learn.predict(img)

    # assume vocab is something like ['hotdog', 'legs'] (check order)
    vocab = list(learn.dls.vocab)
    prob_dict = {label: float(probs[i]) for i, label in enumerate(vocab)}

    return {
        "pred_label": str(pred_class),
        "probs": prob_dict,
    }
