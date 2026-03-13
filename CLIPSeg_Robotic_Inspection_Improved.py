import subprocess, sys
pkgs = [
    "transformers>=4.38.2", "pycocotools", "roboflow",
    "scikit-learn", "matplotlib", "tqdm", "pyyaml", "torchvision",
]
for pkg in pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("Packages ready.")


import os, random, time, warnings, yaml, json, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image as PILImage
import torchvision.transforms.functional as TF
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("CLIPSeg")

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    log.info("Apple MPS detected.")
else:
    device = torch.device("cpu")
    log.info("Running on CPU.")
print(f"Device: {device}")


# ── Hardcoded dataset paths (same as original notebook) ──────────────────
CRACKS_DIR  = Path("/Users/apple/Desktop/origin/cracks.v1i.coco")
DRYWALL_DIR = Path("/Users/apple/Desktop/origin/Drywall-Join-Detect.v2i.coco")

# ── Hyper-parameters ─────────────────────────────────────────────────────
IMG_SIZE   = 352
EPOCHS     = 10
BATCH_SIZE = 4
ACCUM_STEPS = 4          # effective batch = 4 * 4 = 16
LR         = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
PATIENCE   = 3
USE_AMP    = (device.type == "cuda")

CHECKPOINT_DIR = Path("./checkpoints"); CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR     = Path("./outputs");     OUTPUT_DIR.mkdir(exist_ok=True)

PROMPTS = ["segment crack", "cracks in concrete", "damaged surface"]
PRIMARY_PROMPT = PROMPTS[0]

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MORPH_CLOSE_K = 5
MIN_COMPONENT_AREA = 100

print(f"Cracks  dir : {CRACKS_DIR}  | exists={CRACKS_DIR.exists()}")
print(f"Drywall dir : {DRYWALL_DIR} | exists={DRYWALL_DIR.exists()}")
print(f"Image size  : {IMG_SIZE}x{IMG_SIZE}")


from pycocotools.coco import COCO

def check_dataset(root: Path) -> bool:
    """Verify at least one annotation file exists."""
    anns = list(root.rglob("_annotations.coco.json")) if root.exists() else []
    if anns:
        log.info(f"Dataset OK: {root}")
        return True
    log.error(f"Dataset MISSING: {root}. Check path in Cell 2.")
    return False

def dataset_stats(root: Path, split: str) -> dict:
    """Compute crack-pixel ratio for a given split (samples first 50 images)."""
    ann_file = root / split / "_annotations.coco.json"
    if not ann_file.exists():
        return {}
    coco = COCO(str(ann_file))
    ids  = coco.getImgIds()
    total_px = total_fg = 0
    for img_id in ids[:50]:
        info = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for ann in anns:
            if "segmentation" in ann and ann["segmentation"]:
                mask = np.maximum(mask, coco.annToMask(ann))
        total_px += info["height"] * info["width"]
        total_fg += int(mask.sum())
    pct = 100 * total_fg / total_px if total_px else 0
    return {"n_images": len(ids), "crack_pct": round(pct, 3)}

cracks_ok  = check_dataset(CRACKS_DIR)
drywall_ok = check_dataset(DRYWALL_DIR)

if not cracks_ok:
    raise RuntimeError(
        f"Cracks dataset not found at {CRACKS_DIR}. "
        "Update CRACKS_DIR in Cell 2."
    )

print()
for split in ["train", "valid", "test"]:
    s = dataset_stats(CRACKS_DIR, split)
    if s:
        print(f"Cracks/{split}: {s['n_images']:>5} images | "
              f"crack pixel ratio: {s['crack_pct']:.3f}%")
for split in ["train", "valid"]:
    s = dataset_stats(DRYWALL_DIR, split)
    if s:
        print(f"Drywall/{split}: {s['n_images']:>5} images | "
              f"defect pixel ratio: {s['crack_pct']:.3f}%")


# We use PyTorch / torchvision / PIL for ALL transforms to avoid
# OpenCV library conflicts present in some environments.

def train_transform(image_np: np.ndarray, mask_np: np.ndarray):
    """Rich augmentation pipeline using PIL + torchvision only."""
    from PIL import ImageFilter
    img = PILImage.fromarray(image_np).resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
    msk = PILImage.fromarray(mask_np, mode="L").resize((IMG_SIZE, IMG_SIZE), PILImage.NEAREST)

    # Random horizontal flip
    if random.random() > 0.5:
        img, msk = TF.hflip(img), TF.hflip(msk)

    # Random vertical flip
    if random.random() > 0.5:
        img, msk = TF.vflip(img), TF.vflip(msk)

    # Random 90-degree rotation
    if random.random() > 0.5:
        k = random.choice([1, 2, 3])
        img = img.rotate(90 * k)
        msk = msk.rotate(90 * k)

    # Brightness + Contrast jitter
    if random.random() > 0.5:
        img = TF.adjust_brightness(img, 1.0 + (random.random() - 0.5) * 0.6)
        img = TF.adjust_contrast(img,   1.0 + (random.random() - 0.5) * 0.6)

    # Gaussian blur (simulate focus variation)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Auto-contrast (CLAHE-like)
    if random.random() > 0.7:
        img = TF.autocontrast(img)

    # Sharpen (crack edge enhancement)
    if random.random() > 0.7:
        img = TF.adjust_sharpness(img, sharpness_factor=random.uniform(1.5, 3.0))

    return np.array(img, dtype=np.uint8), np.array(msk, dtype=np.uint8)


def val_transform(image_np: np.ndarray, mask_np: np.ndarray):
    """Validation/test: resize only."""
    img = PILImage.fromarray(image_np).resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
    msk = PILImage.fromarray(mask_np, mode="L").resize((IMG_SIZE, IMG_SIZE), PILImage.NEAREST)
    return np.array(img, dtype=np.uint8), np.array(msk, dtype=np.uint8)

print("PIL-based augmentation pipelines ready.")


from torch.utils.data import Dataset, DataLoader

def extract_mask_from_coco(coco: COCO, img_info: dict) -> np.ndarray:
    """Build a binary uint8 mask from COCO polygon/bbox annotations."""
    ann_ids = coco.getAnnIds(imgIds=img_info["id"])
    anns    = coco.loadAnns(ann_ids)
    mask    = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
    for ann in anns:
        if "segmentation" in ann and ann["segmentation"]:
            mask = np.maximum(mask, coco.annToMask(ann))
        else:
            x, y, w, h = [int(v) for v in ann["bbox"]]
            mask[y:y+h, x:x+w] = 1
    return mask.clip(0, 1).astype(np.uint8)


class InspectionDataset(Dataset):
    """COCO segmentation dataset for CLIPSeg.

    Args:
        root_dir:  Dataset root (contains train/valid/test subfolders).
        split:     Split subfolder name.
        prompt:    Natural-language segmentation prompt.
        processor: CLIPSegProcessor instance.
        transform: Callable(image_np, mask_np) -> (image_np, mask_np).
    """
    def __init__(self, root_dir, split, prompt, processor, transform=None):
        self.root_dir  = Path(root_dir)
        self.split     = split
        self.prompt    = prompt
        self.processor = processor
        self.transform = transform
        self.img_dir   = self.root_dir / split
        ann_file       = self.img_dir / "_annotations.coco.json"
        if ann_file.exists():
            self.coco    = COCO(str(ann_file))
            self.img_ids = self.coco.getImgIds()
        else:
            self.coco    = None
            self.img_ids = []

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info["file_name"]

        # Load via PIL -- avoids cv2 library conflicts
        try:
            image = np.array(PILImage.open(str(img_path)).convert("RGB"),
                             dtype=np.uint8)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        mask = extract_mask_from_coco(self.coco, img_info)

        if self.transform:
            image, mask = self.transform(image, mask)

        inputs = self.processor(
            text=self.prompt, images=image,
            return_tensors="pt", padding=True
        )
        inputs   = {k: v.squeeze(0) for k, v in inputs.items()}
        mask_t   = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return inputs, mask_t, img_info["file_name"]


print("Dataset class ready.")


from transformers import CLIPSegProcessor

# use_fast=False suppresses the ViTImageProcessor deprecation warning
processor = CLIPSegProcessor.from_pretrained(
    "CIDAS/clipseg-rd64-refined", use_fast=False
)

train_ds = InspectionDataset(CRACKS_DIR, "train", PRIMARY_PROMPT, processor, train_transform)
val_ds   = InspectionDataset(CRACKS_DIR, "valid", PRIMARY_PROMPT, processor, val_transform)
test_ds  = InspectionDataset(CRACKS_DIR, "test",  PRIMARY_PROMPT, processor, val_transform)

for name, ds in [("train", train_ds), ("valid", val_ds), ("test", test_ds)]:
    print(f"{name:>6}: {len(ds)} samples")

if len(train_ds) == 0:
    raise RuntimeError("Training set is empty -- check dataset path/structure.")


def denorm(tensor):
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std  = np.array([0.26862954, 0.26130258, 0.27577711])
    return np.clip(tensor.permute(1,2,0).numpy() * std + mean, 0, 1)

def visualise_samples(ds, n=4, title=""):
    """Show n random images alongside their ground-truth masks.

    Args:
        ds:    InspectionDataset instance.
        n:     Number of random samples to display.
        title: Figure super-title string.
    """
    import matplotlib.pyplot as plt
    if ds is None or len(ds) == 0:
        print("No samples to display."); return
    idxs = random.sample(range(len(ds)), min(n, len(ds)))
    fig, axes = plt.subplots(n, 2, figsize=(8, 4*n))
    if n == 1: axes = [axes]
    for row, idx in enumerate(idxs):
        inp, msk, fname = ds[idx]
        axes[row][0].imshow(denorm(inp["pixel_values"]))
        axes[row][0].set_title(f"Image: {fname}", fontsize=8)
        axes[row][0].axis("off")
        axes[row][1].imshow(msk[0].numpy(), cmap="gray")
        axes[row][1].set_title("GT Mask")
        axes[row][1].axis("off")
    fig.suptitle(title, fontsize=13); plt.tight_layout(); plt.show()

visualise_samples(train_ds, n=4, title="Cracks Training Samples + GT Masks")


from transformers import CLIPSegForImageSegmentation

model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Freeze CLIP backbone (vision encoder + text encoder)
for param in model.clip.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Total params    : {total:,}")
print(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")
model.to(device);


class FocalBCELoss(nn.Module):
    """Focal Binary Cross Entropy: down-weights easy negatives."""
    def __init__(self, gamma=2.0, pos_weight=10.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits, targets):
        pw  = self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none")
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


class DiceLoss(nn.Module):
    def forward(self, logits, targets, smooth=1e-6):
        p     = torch.sigmoid(logits)
        inter = (p * targets).sum(dim=(2, 3))
        union = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        return (1 - (2*inter + smooth) / (union + smooth)).mean()


class TverskyLoss(nn.Module):
    """Tversky loss: penalises false negatives more than false positives."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        p  = torch.sigmoid(logits)
        tp = (p * targets).sum(dim=(2,3))
        fp = (p * (1-targets)).sum(dim=(2,3))
        fn = ((1-p) * targets).sum(dim=(2,3))
        t  = (tp+self.smooth)/(tp+self.alpha*fp+self.beta*fn+self.smooth)
        return (1 - t).mean()


class ComboLoss(nn.Module):
    """Focal-BCE + Dice combination loss."""
    def __init__(self, bce_w=0.5, dice_w=0.5, gamma=2.0, pos_weight=10.0):
        super().__init__()
        self.focal  = FocalBCELoss(gamma, pos_weight)
        self.dice   = DiceLoss()
        self.bce_w  = bce_w
        self.dice_w = dice_w

    def forward(self, logits, targets):
        return self.bce_w*self.focal(logits, targets) + \
               self.dice_w*self.dice(logits, targets)


criterion = ComboLoss(bce_w=0.5, dice_w=0.5, gamma=2.0, pos_weight=10.0)
print("Loss functions initialised.")


import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

def run_training(model, train_ds, val_ds):
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=max(1, len(train_loader)//ACCUM_STEPS),
        epochs=EPOCHS, pct_start=WARMUP_EPOCHS/EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)

    tr_losses, vl_losses, vl_dices, vl_ious = [], [], [], []
    best_dice, patience_ctr = 0.0, 0
    best_ckpt = CHECKPOINT_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS+1):
        # ---- Train -------------------------------------------------------
        model.train()
        ep_loss = 0.0
        optimizer.zero_grad()
        for i, (inp, masks, _) in enumerate(
                tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} Train")):
            inp   = {k: v.to(device) for k, v in inp.items()}
            masks = masks.to(device)
            with autocast(enabled=USE_AMP):
                logits = model(**inp).logits.unsqueeze(1)
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
                loss   = criterion(logits, masks) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i+1) % ACCUM_STEPS == 0 or (i+1) == len(train_loader):
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(); scheduler.step()
            ep_loss += loss.item() * ACCUM_STEPS
        mean_tr = ep_loss / len(train_loader)
        tr_losses.append(mean_tr)

        # ---- Validate ----------------------------------------------------
        model.eval()
        vl, vd, vi = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inp, masks, _ in tqdm(
                    val_loader, desc=f"Ep {epoch}/{EPOCHS} Val", leave=False):
                inp   = {k: v.to(device) for k, v in inp.items()}
                masks = masks.to(device)
                logits = model(**inp).logits.unsqueeze(1)
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
                vl += criterion(logits, masks).item()
                probs = torch.sigmoid(logits).cpu().numpy()
                gts   = masks.cpu().numpy().astype(bool)
                for j in range(len(probs)):
                    pred  = probs[j,0] > 0.5; gt = gts[j,0]
                    inter = np.logical_and(pred, gt).sum()
                    union = np.logical_or(pred,  gt).sum()
                    denom = pred.sum() + gt.sum()
                    vi += inter/union if union > 0 else 0
                    vd += 2*inter/denom if denom > 0 else 0

        n = len(val_ds)
        mean_vl = vl/len(val_loader)
        mean_vd = vd/n; mean_vi = vi/n
        vl_losses.append(mean_vl); vl_dices.append(mean_vd); vl_ious.append(mean_vi)

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d} | Train: {mean_tr:.4f} | Val: {mean_vl:.4f} "
              f"| Dice: {mean_vd:.4f} | IoU: {mean_vi:.4f} | LR: {lr_now:.2e}")

        if mean_vd > best_dice:
            best_dice = mean_vd
            torch.save(model.state_dict(), best_ckpt)
            log.info(f"  => Checkpoint saved  (Dice={best_dice:.4f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"Early stopping at epoch {epoch}.")
                break

    return tr_losses, vl_losses, vl_dices, vl_ious

tr_losses, vl_losses, vl_dices, vl_ious = run_training(model, train_ds, val_ds)


import matplotlib.pyplot as plt

def plot_curves(tr_l, vl_l, vl_d, vl_i):
    if not tr_l:
        print("No training history."); return
    eps = range(1, len(tr_l)+1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(eps, tr_l, label="Train"); axes[0].plot(eps, vl_l, label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(eps, vl_d, color="green"); axes[1].set_title("Val Dice / Epoch")
    axes[2].plot(eps, vl_i, color="orange"); axes[2].set_title("Val IoU / Epoch")
    for ax in axes: ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=120)
    plt.show()

plot_curves(tr_losses, vl_losses, vl_dices, vl_ious)


import cv2  # only used for morphological ops – not for image loading
from scipy.ndimage import label as scipy_label
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_curve, auc, confusion_matrix)

# Load best checkpoint if available
best_ckpt = CHECKPOINT_DIR / "best_model.pt"
if best_ckpt.exists():
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    log.info(f"Loaded checkpoint: {best_ckpt}")

def sliding_window_infer(model, processor, img_np, prompt,
                          tile=352, stride=256):
    """Tiled inference -- averages overlapping probability maps."""
    H, W = img_np.shape[:2]
    prob  = np.zeros((H, W), dtype=np.float32)
    cnt   = np.zeros((H, W), dtype=np.float32)
    ys = sorted(set(list(range(0,H-tile+1,stride)) + ([H-tile] if H>tile else [0])))
    xs = sorted(set(list(range(0,W-tile+1,stride)) + ([W-tile] if W>tile else [0])))
    model.eval()
    for y in ys:
        for x in xs:
            crop = img_np[y:y+tile, x:x+tile]
            inp  = processor(text=prompt, images=crop,
                              return_tensors="pt", padding=True)
            inp  = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                logit = model(**inp).logits.unsqueeze(1)
                logit = F.interpolate(logit, size=(tile,tile),
                                      mode="bilinear", align_corners=False)
                p = torch.sigmoid(logit).cpu().numpy()[0,0]
            prob[y:y+tile, x:x+tile] += p
            cnt[y:y+tile,  x:x+tile] += 1
    return prob / np.maximum(cnt, 1)


def ensemble_prompts(model, processor, img_np, prompts, tile=352):
    """Average probability maps from multiple prompts."""
    maps = [sliding_window_infer(model, processor, img_np, p, tile)
            for p in prompts]
    return np.mean(maps, axis=0)


def post_process(prob, thr=0.5, close_k=5, min_area=100):
    """Threshold + morphological closing + small-component removal."""
    binary = (prob > thr).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.dilate(binary, kernel, iterations=1)
    labeled, n = scipy_label(binary)
    for comp in range(1, n+1):
        if (labeled == comp).sum() < min_area:
            binary[labeled == comp] = 0
    return binary.astype(np.uint8)


def evaluate(model, processor, ds, prompts, thresholds=THRESHOLDS):
    res = {t: {"iou":[],"dice":[],"prec":[],"rec":[],"f1":[]}
           for t in thresholds}
    all_probs, all_gts = [], []

    for idx in tqdm(range(len(ds)), desc="Evaluating"):
        t0 = time.time()
        inp, msk_t, fname = ds[idx]
        img_np = np.clip(
            (inp["pixel_values"].permute(1,2,0).numpy() *
             np.array([0.26862954,0.26130258,0.27577711]) +
             np.array([0.48145466,0.4578275,0.40821073])) * 255,
            0, 255).astype(np.uint8)
        gt = msk_t[0].numpy().astype(bool)

        prob_map = ensemble_prompts(model, processor, img_np, prompts)
        elapsed  = (time.time()-t0)*1000
        all_probs.append(prob_map.ravel())
        all_gts.append(gt.ravel())

        for t in thresholds:
            pred  = post_process(prob_map, t, MORPH_CLOSE_K, MIN_COMPONENT_AREA).astype(bool)
            inter = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            denom = pred.sum() + gt.sum()
            res[t]["iou"].append(inter/union if union>0 else 0)
            res[t]["dice"].append(2*inter/denom if denom>0 else 0)
            res[t]["prec"].append(precision_score(gt.ravel(), pred.ravel(), zero_division=0))
            res[t]["rec"].append(recall_score(gt.ravel(), pred.ravel(), zero_division=0))
            res[t]["f1"].append(f1_score(gt.ravel(), pred.ravel(), zero_division=0))

        if idx % 20 == 0:
            log.info(f"  [{idx}/{len(ds)}]  {elapsed:.1f} ms/image")

        # Save green overlay
        final   = post_process(prob_map, 0.5, MORPH_CLOSE_K, MIN_COMPONENT_AREA)
        overlay = img_np.copy()
        overlay[final==1] = [0,220,0]
        overlay = (img_np * 0.6 + overlay * 0.4).astype(np.uint8)
        PILImage.fromarray(overlay).save(
            OUTPUT_DIR / f"{Path(fname).stem}_pred.png")

    # ---- Results --------------------------------------------------------
    print("\n--- Threshold Tuning Results ---")
    best_t, best_dice = 0.5, 0.0
    for t in thresholds:
        m = {k: np.mean(v) for k,v in res[t].items()}
        print(f"Thr {t:.1f} | IoU:{m['iou']:.4f} Dice:{m['dice']:.4f} "
              f"P:{m['prec']:.4f} R:{m['rec']:.4f} F1:{m['f1']:.4f}")
        if m["dice"] > best_dice:
            best_dice, best_t = m["dice"], t
    print(f"\nBest threshold: {best_t}  (Dice={best_dice:.4f})")

    # ROC curve
    ap = np.concatenate(all_probs); ag = np.concatenate(all_gts)
    fpr, tpr, _ = roc_curve(ag, ap)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Pixel-Level ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"roc_curve.png", dpi=120); plt.show()

    # Confusion matrix
    pb = (ap > best_t).astype(int); gb = ag.astype(int)
    cm = confusion_matrix(gb, pb)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Bg","Crack"]); ax.set_yticklabels(["Bg","Crack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    plt.title(f"Confusion Matrix  thr={best_t}"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"confusion_matrix.png", dpi=120); plt.show()

    return res, best_t

eval_results, best_thr = evaluate(model, processor, val_ds, PROMPTS)


def visualise_predictions(model, processor, ds, prompts, n=4, thr=0.5):
    """5-panel gallery: Original / GT / Prob Heatmap / Prediction / Error Map."""
    if ds is None or len(ds) == 0:
        print("No data."); return
    idxs = random.sample(range(len(ds)), min(n, len(ds)))
    fig, axes = plt.subplots(n, 5, figsize=(22, 4*n))
    if n == 1: axes = [axes]
    titles = ["Original", "GT Mask", "Prob Heatmap", "Post-Proc Pred", "Error (R=FP B=FN)"]
    for col, t in enumerate(titles):
        axes[0][col].set_title(t, fontsize=10, fontweight="bold")

    for row, idx in enumerate(idxs):
        inp, msk_t, fname = ds[idx]
        img_np = np.clip(
            (inp["pixel_values"].permute(1,2,0).numpy() *
             np.array([0.26862954,0.26130258,0.27577711]) +
             np.array([0.48145466,0.4578275,0.40821073])) * 255,
            0, 255).astype(np.uint8)
        gt  = msk_t[0].numpy().astype(bool)

        prob_map = ensemble_prompts(model, processor, img_np, prompts)
        pred     = post_process(prob_map, thr, MORPH_CLOSE_K,
                                 MIN_COMPONENT_AREA).astype(bool)

        err = np.zeros((*gt.shape, 3), dtype=np.uint8)
        err[np.logical_and(pred, ~gt)] = [255, 0,   0]   # FP red
        err[np.logical_and(~pred, gt)] = [0,   0, 255]   # FN blue

        axes[row][0].imshow(img_np)
        axes[row][1].imshow(gt, cmap="gray")
        axes[row][2].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
        axes[row][3].imshow(pred, cmap="gray")
        axes[row][4].imshow(err)
        for col in range(5): axes[row][col].axis("off")

    plt.suptitle("CLIPSeg Prediction Gallery", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"prediction_gallery.png", dpi=120, bbox_inches="tight")
    plt.show()

visualise_predictions(model, processor, val_ds, PROMPTS, n=4, thr=best_thr)

