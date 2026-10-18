from dataclasses import dataclass
from ultralytics import YOLO
from ..utils.config import load_classes

CLASS_NAMES = load_classes()["detection"]   # {0: crater, 1: rock}

@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

class LunarDetector:
    """YOLOv8s crater + rock detector. Input: BGR uint8 image (OpenCV / cv_bridge 'bgr8')."""
    def __init__(self, weights, conf=0.25, iou=0.5, imgsz=640, device=None):
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device

    def detect(self, bgr, classes=None):
        r = self.model.predict(bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                               classes=classes, device=self.device, verbose=False)[0]
        out = []
        for b in r.boxes:
            c = int(b.cls)
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append(Detection(c, CLASS_NAMES[c], float(b.conf), x1, y1, x2, y2))
        return out

    def detect_craters(self, bgr):
        return self.detect(bgr, classes=[0])
