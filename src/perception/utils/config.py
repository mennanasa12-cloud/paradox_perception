from pathlib import Path
import yaml
from .paths import PROJECT_ROOT

def load_classes(path=PROJECT_ROOT / "configs" / "classes.yaml"):
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)