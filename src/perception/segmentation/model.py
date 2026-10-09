import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

def build_model(num_classes=5, pretrained="nvidia/segformer-b0-finetuned-ade-512-512"):
    return SegformerForSemanticSegmentation.from_pretrained(
        pretrained, num_labels=num_classes, ignore_mismatched_sizes=True)

def forward_logits(model, x):
    """x: normalized tensor (B,3,H,W) -> logits (B,C,H,W)."""
    logits = model(pixel_values=x).logits
    return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

@torch.no_grad()
def predict(model, x):
    """Class-ID mask (B,H,W). This is the function the ROS 2 node will call."""
    model.eval()
    return forward_logits(model, x).argmax(1)
