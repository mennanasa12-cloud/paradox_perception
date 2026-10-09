import json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..dataset.lusnar_seg import LuSNARSeg
from .model import build_model, forward_logits

def confusion(pred, target, n, ignore=255):
    k = target != ignore
    return torch.bincount(n * target[k] + pred[k], minlength=n * n).reshape(n, n)

def scores(cm):
    inter = cm.diag().float()
    union = cm.sum(0).float() + cm.sum(1).float() - inter
    iou = inter / union.clamp(min=1)
    present = cm.sum(1) > 0            # mIoU over classes that appear in the ground truth
    return iou, iou[present].mean().item()

@torch.no_grad()
def evaluate(model, loader, n, ignore, device="cuda"):
    model.eval()
    cm = torch.zeros(n, n, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = forward_logits(model, x).argmax(1)
        cm += confusion(pred, y, n, ignore)
    return scores(cm.cpu())

def train(data_root, cfg, out_dir, epochs=30, bs=8, lr=6e-5, workers=2):
    device = "cuda"
    n = len(cfg["segmentation"])
    ignore = cfg["segmentation_ignore_index"]
    names = [cfg["segmentation"][i] for i in range(n)]
    w = torch.tensor([cfg["segmentation_class_weights"][i] for i in range(n)],
                     dtype=torch.float32, device=device)

    tr = DataLoader(LuSNARSeg(data_root, "train"), bs, shuffle=True,
                    num_workers=workers, drop_last=True, pin_memory=True)
    va = DataLoader(LuSNARSeg(data_root, "val"), bs, num_workers=workers)

    model = build_model(n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr))
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=ignore)
    scaler = torch.amp.GradScaler("cuda")

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(forward_logits(model, x).float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        iou, miou = evaluate(model, va, n, ignore, device)
        row = {"epoch": ep, "loss": round(tot / len(tr), 4), "val_mIoU": round(miou, 4),
               "val_IoU": {names[i]: round(iou[i].item(), 3) for i in range(n)}}
        log.append(row); print(f"{time.time()-t0:.0f}s", row)
        if miou > best:
            best = miou
            torch.save(model.state_dict(), out / "best.pt")
        json.dump(log, open(out / "log.json", "w"), indent=2)
    print("best val mIoU:", round(best, 4))
    return model

def test(data_root, cfg, ckpt, bs=8):
    n = len(cfg["segmentation"]); ignore = cfg["segmentation_ignore_index"]
    names = [cfg["segmentation"][i] for i in range(n)]
    model = build_model(n).to("cuda")
    model.load_state_dict(torch.load(ckpt))
    te = DataLoader(LuSNARSeg(data_root, "test"), bs, num_workers=2)
    iou, miou = evaluate(model, te, n, ignore)
    print("TEST mIoU:", round(miou, 4), {names[i]: round(iou[i].item(), 3) for i in range(n)})
