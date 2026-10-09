from pathlib import Path
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def get_transforms(train: bool):
    norm = [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    if not train:
        return A.Compose(norm)
    # horizontal flip only (sky is always on top); no hue shifts (the Moon is nearly colorless)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
    ] + norm)

class LuSNARSeg:
    """Reads <root>/<split>/images/*.png and masks/*.png (class IDs 0-4, 255=ignore)."""
    def __init__(self, root, split, train=None):
        self.img_dir = Path(root) / split / "images"
        self.mask_dir = Path(root) / split / "masks"
        self.names = sorted(p.name for p in self.img_dir.glob("*.png"))
        self.tf = get_transforms(split == "train" if train is None else train)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = np.array(Image.open(self.img_dir / n).convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / n))
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()
