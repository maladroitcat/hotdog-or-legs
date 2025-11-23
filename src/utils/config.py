from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load a YAML config file from the given path.
    Defaults to 'config.yaml' in the project root.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path.resolve()}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)
