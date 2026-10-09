from pathlib import Path
import yaml

def load_classes(path="configs/classes.yaml"):
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)