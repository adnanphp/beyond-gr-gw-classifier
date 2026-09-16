"""Load and access config.yaml."""
from pathlib import Path
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get(cfg, *keys, default=None):
    for k in keys:
        if isinstance(cfg, dict) and k in cfg:
            cfg = cfg[k]
        else:
            return default
    return cfg
