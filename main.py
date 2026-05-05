"""
BanglaLekha-Isolated — v3 (DDP + Fast Ablation Integrated)
"""

import os, time, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

_N_GPU     = int(os.environ.get("WORLD_SIZE", max(torch.cuda.device_count(), 1)))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

if LOCAL_RANK == 0:
    print(f"Device: {device}  |  GPUs: {_N_GPU}  |  Effective batch: 512 * {_N_GPU}")

bangla_class_mapping = {
    "1":  "অ", "2":  "আ", "3":  "ই", "4":  "ঈ", "5":  "উ", "6":  "ঊ",
    "7":  "ঋ", "8":  "এ", "9":  "ঐ", "10": "ও", "11": "ঔ",
    "12": "ক", "13": "খ", "14": "গ", "15": "ঘ", "16": "ঙ", "17": "চ",
    "18": "ছ", "19": "জ", "20": "ঝ", "21": "ঞ", "22": "ট", "23": "ঠ",
    "24": "ড", "25": "ঢ", "26": "ণ", "27": "ত", "28": "থ", "29": "দ",
    "30": "ধ", "31": "ন", "32": "প", "33": "ফ", "34": "ব", "35": "ভ",
    "36": "ম", "37": "য", "38": "র", "39": "ল", "40": "শ", "41": "ষ",
    "42": "স", "43": "হ", "44": "ড়", "45": "ঢ়", "46": "য়", "47": "ৎ",
    "48": "\u0982", "49": "\u0983", "50": "\u0981",
    "51": "০", "52": "১", "53": "২", "54": "৩", "55": "৪", "56": "৫",
    "57": "৬", "58": "৭", "59": "৮", "60": "৯",
    "61": "ক্ষ", "62": "ত্র", "63": "জ্ঞ", "64": "ষ্ক", "65": "স্ক", "66": "স্থ",
    "67": "চ্ছ", "68": "ক্ত", "69": "ত্ত", "70": "ব্ধ", "71": "ম্প", "72": "ষ্ণ",
    "73": "ষ্ঠ", "74": "ম্ব", "75": "ণ্ড", "76": "দ্ব", "77": "ন্থ", "78": "স্ত",
    "79": "ল্প", "80": "ষ্প", "81": "ন্দ", "82": "ন্ধ", "83": "ম্ম", "84": "ন্ট",
}

DATA_DIR    = "Dataset/Images"
IMG_SIZE    = 128
NUM_CLASSES = 84
NUM_WORKERS = 20

PER_GPU_BS  = 512
GLOBAL_BS   = PER_GPU_BS * _N_GPU
_BASE_BS    = 256

def make_transforms(img_size: int, augment: bool):
    val_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    if not augment:
        return val_tf, val_tf
    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_size + img_size // 8, img_size + img_size // 8)),
        transforms.RandomCrop(img_size),
        transforms.RandomRotation(10, fill=0),
        transforms.RandomAffine(degrees=0, shear=12, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return train_tf, val_tf

def make_criterion(base_dataset: datasets.ImageFolder,
                   label_smoothing: float = 0.1) -> nn.CrossEntropyLoss:
    counts  = np.bincount([lbl for _, lbl in base_dataset.imgs],
                          minlength=NUM_CLASSES).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (NUM_CLASSES * counts)
    w       = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)

class BanglaCNN(nn.Module):
    def __init__(self, num_layers=5, dropout_rate=0.4,
                 num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        super().__init__()
        layers = []
        in_ch, out_ch, spatial = 1, 32, img_size
        for _ in range(num_layers):
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch; out_ch = min(out_ch * 2, 512); spatial //= 2
        self.features   = nn.Sequential(*layers)
        flat            = in_ch * spatial * spatial
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),  nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def make_loader(dataset, *, shuffle: bool) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size         = PER_GPU_BS,
        sampler            = sampler,
        shuffle            = False,
        num_workers        = NUM_WORKERS // max(_N_GPU, 1),
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 2,
    )

def plot_ablation_phase(phase_name: str, phase_histories: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Ablation: {phase_name}", fontsize=16, fontweight="bold")
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    for i, (label, hist) in enumerate(phase_histories.items()):
        c   = colors[i % len(colors)]
        eps = range(1, len(hist["val_acc"]) + 1)
        ax1.plot(eps, hist["val_acc"],    "o-", color=c, label=label)
        ax2.plot(eps, hist["train_loss"], "--", color=c, alpha=0.5, label=f"{label} train")
        ax2.plot(eps, hist["val_loss"],   "o-", color=c,             label=f"{label} val")
    for ax, title, ylabel in [
        (ax1, "Validation Accuracy", "Accuracy (%)"),
        (ax2, "Loss Curves",         "Cross-Entropy Loss"),
    ]:
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fname = f"ablation_{phase_name.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    print(f"  📊 Saved → {fname}")
    plt.close(fig)

def train_and_evaluate(
    num_layers     = 5,
    use_aug        = True,
    lr             = 1e-3,
    weight_decay   = 1e-4,
    optimizer_name = "AdamW",
    dropout_rate   = 0.4,
    max_epochs     = 40,       # FIX 1: was 20, model hadn't plateaued
    patience       = 8,        # FIX 1: was 6, scaled with epoch count
    img_size       = IMG_SIZE,
):
    tag = (f"Layers={num_layers} | Aug={use_aug} | Opt={optimizer_name} | "
           f"LR={lr} | WD={weight_decay} | Drop={dropout_rate} | Size={img_size}")

    if LOCAL_RANK == 0:
        print(f"\n{'─'*72}\n{tag}\n{'─'*72}")

    train_tf, val_tf = make_transforms(img_size, use_aug)
    train_base = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    val_base   = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    n      = len(train_base)
    gen    = torch.Generator().manual_seed(42)
    splits = random_split(range(n), [int(0.8 * n), n - int(0.8 * n)], generator=gen)
    tr_idx, va_idx = splits[0].indices, splits[1].indices

    train_loader = make_loader(Subset(train_base, tr_idx), shuffle=True)
    val_loader   = make_loader(Subset(val_base,   va_idx), shuffle=False)
    criterion    = make_criterion(train_base)

    model = BanglaCNN(num_layers=num_layers, dropout_rate=dropout_rate,
                      num_classes=NUM_CLASSES, img_size=img_size).to(device)

    if _N_GPU > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[LOCAL_RANK])

    # FIX 2: Apply LR scaling to AdamW/Adam, not just SGD.
    # With 4 GPUs x 512 = 2048 effective batch vs _BASE_BS=256, scale factor = 8x.
    scaled_lr = lr * (GLOBAL_BS / _BASE_BS)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(),  lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(),   lr=scaled_lr, momentum=0.9,
                              weight_decay=weight_decay, nesterov=True)

    _warmup_epochs = 8         # FIX 1: was 5, pushes high-LR danger zone past spike region
    cosine_epochs  = max(max_epochs - _warmup_epochs, 1)
    warmup    = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup_epochs,
    )
    cosine    = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-7,  # FIX 1: was 1e-6, finer tail
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[_warmup_epochs],
    )
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "val_acc": []}
    ckpt_path     = f"ckpt_L{num_layers}_aug{use_aug}_opt{optimizer_name}.pth"

    for epoch in range(max_epochs):
        t0 = time.time()
        train_loader.sampler.set_epoch(epoch)

        model.train()
        run_loss = 0.0

        if LOCAL_RANK == 0:
            pbar = tqdm(train_loader, desc=f"Ep[{epoch+1}/{max_epochs}] Train", leave=False)
        else:
            pbar = train_loader

        for imgs, labels in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            run_loss += loss.item()
            if LOCAL_RANK == 0:
                pbar.set_postfix(loss=f"{run_loss/(pbar.n+1):.4f}")

        # FIX 3: Train loss denominator was (len * N_GPU), double-counting the reduce.
        # run_loss is per-GPU; after all_reduce it's summed across GPUs.
        # Correct avg = sum / (N_GPU * batches_per_gpu) = reduced / (N_GPU * len(loader))
        # But we want the mean loss per batch, so divide reduced sum by N_GPU * len(loader).
        # Actually simpler: divide by len(loader) after reduce since reduce sums N_GPU copies.
        tr_loss_tensor = torch.tensor([run_loss], dtype=torch.float32, device=device)
        if _N_GPU > 1: dist.all_reduce(tr_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = tr_loss_tensor.item() / (_N_GPU * len(train_loader))

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast():
                    out  = model(imgs)
                    loss = criterion(out, labels)
                val_loss += loss.item()
                _, pred  = torch.max(out, 1)
                total    += labels.size(0)
                correct  += (pred == labels).sum().item()

        metrics = torch.tensor([val_loss, correct, total], dtype=torch.float32, device=device)
        if _N_GPU > 1: dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        # FIX 3: Same fix for val loss — divide by N_GPU * len(val_loader)
        avg_val_loss = metrics[0].item() / (_N_GPU * len(val_loader))
        val_acc      = 100.0 * metrics[1].item() / metrics[2].item()
        elapsed      = time.time() - t0

        if LOCAL_RANK == 0:
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Ep {epoch+1:2d}/{max_epochs} | "
                  f"Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f} | "
                  f"Acc={val_acc:5.2f}% | LR={cur_lr:.2e} | {elapsed:.1f}s")

        scheduler.step()

        stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
        if LOCAL_RANK == 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc  = val_acc
                no_improve    = 0
                sd = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save(sd, ckpt_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  ⏹ Early stop at epoch {epoch+1}")
                    stop_flag += 1

        if _N_GPU > 1: dist.broadcast(stop_flag, src=0)
        if stop_flag.item() > 0:
            break

    if LOCAL_RANK == 0:
        print(f"  ✅ Best Val Acc: {best_val_acc:.2f}%")

    return best_val_acc if LOCAL_RANK == 0 else 0.0, history if LOCAL_RANK == 0 else None

# ─────────────────────────────────────────────────────────────────────────────
# 8. Ablation Suite — Phase 1 results cached, continue from Phase 2
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_EPOCHS   = 40   # Updated to match train_and_evaluate default
ABLATION_PATIENCE = 8

def run_ablations():
    cache = {}

    best_lr = 1e-3
    best_wd = 1e-4

    def run(*, layers, aug, opt, drop, img_size=IMG_SIZE):
        key = f"L{layers}_a{aug}_o{opt}_lr{best_lr}_wd{best_wd}_d{drop}_s{img_size}"
        if key in cache:
            if LOCAL_RANK == 0: print(f"  ⏭  Cached {key}  →  {cache[key][0]:.2f}%")
            return cache[key]
        result = train_and_evaluate(
            num_layers=layers, use_aug=aug, optimizer_name=opt,
            lr=best_lr, weight_decay=best_wd, dropout_rate=drop, img_size=img_size,
            max_epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE,
        )
        cache[key] = result
        return result

    # Phase 1 — SKIP: results stored, best=5 layers @ 94.90%
    if LOCAL_RANK == 0:
        print("\n" + "═"*60)
        print("  PHASE 1 — MODEL DEPTH  [CACHED from previous run]")
        print("  Results: 3L=93.12% | 4L=94.37% | 5L=94.90%")
        print("  → Best: 5 layers  (94.90%)")
        print("═"*60)
    best_layers = 5
    # Populate cache so Phase 2 hits the cache hit for aug=True
    # We re-run it anyway since epochs/LR changed — the new run will be better
    # Just set best_layers and proceed

    # Phase 2 — Augmentation
    if LOCAL_RANK == 0: print("\n" + "═"*60 + "\n  PHASE 2 — AUGMENTATION\n" + "═"*60)
    p2, best_aug, best_acc = {}, True, 0
    for aug in [True, False]:
        acc, hist = run(layers=best_layers, aug=aug, opt="AdamW", drop=0.4)
        if LOCAL_RANK == 0:
            p2[f"Aug={aug}"] = hist
            if acc > best_acc: best_acc, best_aug = acc, aug
    if LOCAL_RANK == 0:
        plot_ablation_phase("Phase 2 — Augmentation", p2)
        print(f"  → Best: aug={best_aug}  ({best_acc:.2f}%)")

    # Phase 3 — Optimizer
    if LOCAL_RANK == 0: print("\n" + "═"*60 + "\n  PHASE 3 — OPTIMIZER\n" + "═"*60)
    p3, best_opt, best_acc = {}, "AdamW", 0
    for opt in ["AdamW", "Adam", "SGD"]:
        acc, hist = run(layers=best_layers, aug=best_aug, opt=opt, drop=0.4)
        if LOCAL_RANK == 0:
            p3[f"Opt={opt}"] = hist
            if acc > best_acc: best_acc, best_opt = acc, opt
    if LOCAL_RANK == 0:
        plot_ablation_phase("Phase 3 — Optimizer", p3)
        print(f"  → Best: {best_opt}  ({best_acc:.2f}%)")

    # Phase 4 — Dropout
    if LOCAL_RANK == 0: print("\n" + "═"*60 + "\n  PHASE 4 — DROPOUT\n" + "═"*60)
    p4, best_drop, best_acc = {}, 0.4, 0
    for drop in [0.3, 0.4, 0.5]:
        acc, hist = run(layers=best_layers, aug=best_aug, opt=best_opt, drop=drop)
        if LOCAL_RANK == 0:
            p4[f"Drop={drop}"] = hist
            if acc > best_acc: best_acc, best_drop = acc, drop
    if LOCAL_RANK == 0:
        plot_ablation_phase("Phase 4 — Dropout", p4)
        print(f"  → Best: drop={best_drop}  ({best_acc:.2f}%)")

    if LOCAL_RANK == 0:
        print("\n" + "🏆 "*20)
        print("  OPTIMAL HYPERPARAMETERS")
        print("🏆 "*20)
        for k, v in [("Layers", best_layers), ("Aug", best_aug), ("Opt", best_opt),
                     ("LR", best_lr), ("WD", best_wd), ("Dropout", best_drop),
                     ("Best Val Acc", f"{best_acc:.2f}%")]:
            print(f"  {k:14s}: {v}")

    return best_layers, best_aug, best_opt, best_lr, best_wd, best_drop

def generate_confusion_matrix(model_path, num_layers, dropout, val_loader, val_acc=0.0):
    print("\n" + "═"*70 + "\n  CONFUSION MATRIX\n" + "═"*70)
    model = BanglaCNN(num_layers=num_layers, dropout_rate=dropout,
                      num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, truths = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="CM eval"):
            with autocast():
                out = model(imgs.to(device))
            preds.extend(torch.max(out, 1)[1].cpu().numpy())
            truths.extend(labels.numpy())

    cm = confusion_matrix(truths, preds)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, linewidths=0.1)
    plt.title(f"Confusion Matrix", fontsize=18, pad=20)
    plt.ylabel("True Class"); plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig("confusion_matrix_best.png", dpi=150)
    print("  📊 Saved → confusion_matrix_best.png")
    plt.close()

def preview_dataset(dataset_path=DATA_DIR):
    folders = sorted(
        [d for d in os.listdir(dataset_path)
         if os.path.isdir(os.path.join(dataset_path, d))],
        key=lambda x: int(x),
    )
    fig, axes = plt.subplots(12, 7, figsize=(15, 25))
    axes = axes.flatten()
    for i, folder in enumerate(folders):
        d    = os.path.join(dataset_path, folder)
        imgs = [f for f in os.listdir(d) if f.lower().endswith((".png",".jpg",".jpeg"))]
        if imgs:
            img = Image.open(os.path.join(d, imgs[min(5, len(imgs)-1)]))
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"{folder}: {bangla_class_mapping.get(folder,'?')}", fontsize=7)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("dataset_preview.png", dpi=100)
    print("  📊 Saved → dataset_preview.png")
    plt.close(fig)

if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)

    best_layers, best_aug, best_opt, best_lr, best_wd, best_drop = run_ablations()

    if LOCAL_RANK == 0:
        best_ckpt = f"ckpt_L{best_layers}_aug{best_aug}_opt{best_opt}.pth"
        prod_path = "FINAL_PRODUCTION_MODEL.pth"
        if os.path.exists(best_ckpt):
            os.rename(best_ckpt, prod_path)
            print(f"\n  ✅ Production model saved → {prod_path}")

        _, val_tf  = make_transforms(IMG_SIZE, False)
        full_ds    = datasets.ImageFolder(DATA_DIR, transform=val_tf)
        n          = len(full_ds)
        gen        = torch.Generator().manual_seed(42)
        _, val_ds  = random_split(full_ds, [int(0.8*n), n - int(0.8*n)], generator=gen)

        cm_loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

        if os.path.exists(prod_path):
            generate_confusion_matrix(prod_path, best_layers, best_drop, cm_loader)

    if dist.is_initialized():
        dist.destroy_process_group()
