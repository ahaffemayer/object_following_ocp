import yaml
from pathlib import Path

def load_config(path_to_yaml: str) -> dict:
    path = Path(path_to_yaml)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
