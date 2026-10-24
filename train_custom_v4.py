import os
import sys
import time
import copy
import json
import random
import argparse
import warnings
from io import BytesIO

warnings.filterwarnings('ignore', message='.*Palette images with Transparency.*')
warnings.filterwarnings('ignore', message='.*xFormers is not available.*')

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
from datasets import load_dataset

from model_custom import AIForensicNetV4


# ==============================================================================
# 1. FORENSIC NATIVE-RESOLUTION & ANTI-COMPRESSION AUGMENTATION
# ==============================================================================

class RandomJPEGCompression:
    """
    Simulasi kompresi lossy JPEG acak pada skala resolusi asli.
    Melatih model agar kebal terhadap artefak kompresi media sosial (WhatsApp,
    WebP, Instagram, dll.) dan tidak salah mengira artefak kompresi sebagai artefak AI.
    """
    def __init__(self, quality_range=(60, 95), p=0.45):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            return Image.open(buf).convert('RGB')
        return img


class NativeResolutionCrop:
    """
    Penyempurnaan Fundamental V4:
    Mempertahankan resolusi murni citra (tanpa bilinear downsampling global yang
    meredam sinyal frekuensi tinggi). Citra hanya di-crop secara acak pada ukuran
    target (224x224), atau di-pad/upscale minimal hanya jika dimensi asli < 224.
    """
    def __init__(self, size=224, is_train=True):
        self.size = size
        self.is_train = is_train

    def __call__(self, img):
        w, h = img.size
        # Jika dimensi lebih kecil dari ukuran crop, sesuaikan proporsional
        if min(w, h) < self.size:
            scale = float(self.size) / float(min(w, h))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            w, h = img.size

        if self.is_train:
            # Random Crop untuk pelatihan
            left = random.randint(0, w - self.size)
            top = random.randint(0, h - self.size)
        else:
            # Center Crop deterministik untuk evaluasi
            left = (w - self.size) // 2
            top = (h - self.size) // 2

        return img.crop((left, top, left + self.size, top + self.size))


class HFToPyTorchDataset(Dataset):
    """Adapter untuk menghubungkan Hugging Face Dataset dengan PyTorch DataLoader."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_v4_transforms(img_size=224):
    """
    Pipeline Transformasi AIForensicNet V4:
    - Kompresi JPEG acak pada resolusi asli.
    - Native Resolution Random Crop (menjaga artefak grid generator tetap utuh).
    - Random Horizontal Flip (tanpa rotasi bersudut hitam).
    - Gaussian blur mikro dan Color Jitter sangat halus.
    """
    train_transforms = transforms.Compose([
        RandomJPEGCompression(quality_range=(60, 95), p=0.45),
        NativeResolutionCrop(size=img_size, is_train=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.4))
        ], p=0.15),
        transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.06),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transforms = transforms.Compose([
        NativeResolutionCrop(size=img_size, is_train=False),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, eval_transforms


# ==============================================================================
# 2. SEED & REPRODUCIBILITY SETUP
# ==============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ==============================================================================
# 3. WORKFLOW PELATIHAN DAN EVALUASI V4
# ==============================================================================

def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("🚀 PELATIHAN AIForensicNet V4 (Tri-Stream NPR + Spectral FFT + Native Crop)")
    print("=" * 75)
    if torch.cuda.is_available():
        print(f"Device Utama        : CUDA ({torch.cuda.get_device_name(0)})")
        print(f"Total Alokasi VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"Device Utama        : CPU")
    print(f"Total Epoch         : {args.epochs}")
    print(f"Batch Size          : {args.batch_size} (Akumulasi: {args.accum_steps} -> Efektif: {args.batch_size * args.accum_steps})")
    print(f"Learning Rate Puncak: {args.lr} (Warmup 2 Ep + Cosine Annealing)")
    print(f"Label Smoothing     : {args.label_smoothing}")
    print(f"Model Checkpoint    : {args.output_path}")
    print("=" * 75)

    # 1. Pembagian Dataset 3-Way Berstrata (80% Train, 10% Val, 10% Test)
    print("\n📦 Memuat dataset dari cache lokal / Hugging Face...")
    raw_dataset = load_dataset(
        args.dataset_name,
        split="train",
        cache_dir=args.cache_dir,
        verification_mode="no_checks"
    )

    # Split 1: 80% Train vs 20% (Val + Test)
    split1 = raw_dataset.train_test_split(test_size=0.20, stratify_by_column="label", seed=args.seed)
    train_hf = split1["train"]
    val_test_hf = split1["test"]

    # Split 2: 10% Val vs 10% Test murni
    split2 = val_test_hf.train_test_split(test_size=0.50, stratify_by_column="label", seed=args.seed)
    val_hf = split2["train"]
    test_hf = split2["test"]

    print(f"Total Keseluruhan Dataset : {len(raw_dataset):,} citra")
    print(f"├─ Data Latih (Train 80%) : {len(train_hf):,} citra")
    print(f"├─ Data Validasi (Val 10%): {len(val_hf):,} citra (digunakan untuk checkpointing)")
    print(f"└─ Data Uji Murni (Test 10%): {len(test_hf):,} citra (disentuh 1x saja di akhir)")

    train_transforms, eval_transforms = get_v4_transforms(img_size=args.img_size)
    train_ds = HFToPyTorchDataset(train_hf, transform=train_transforms)
    val_ds = HFToPyTorchDataset(val_hf, transform=eval_transforms)
    test_ds = HFToPyTorchDataset(test_hf, transform=eval_transforms)

    use_pin = torch.cuda.is_available()
    use_persistent = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=use_pin, persistent_workers=use_persistent)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin, persistent_workers=use_persistent)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin, persistent_workers=use_persistent)

    # 2. Inisialisasi Model AIForensicNetV4
    model = AIForensicNetV4(num_classes=2, dropout_p=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🧠 Arsitektur AIForensicNet V4 Berhasil Diinisialisasi:")
    print(f"├─ Total Parameter    : {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"├─ Trainable Parameter: {trainable_params:,} (~{trainable_params/1e6:.2f}M)")
    print(f"└─ Fitur Kunci        : Tri-Stream (15-ch Multi-Scale + NPR, 2D-FFT Spectral, RGB) + MultiStatPool2d")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = min(2, args.epochs // 4)
    cosine_epochs = max(1, args.epochs - warmup_epochs)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, cosine_epochs))
            return max(1e-3, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    print("\n🏁 Memulai loop pelatihan...")
    for epoch in range(args.epochs):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # ---------------- FASE PELATIHAN ----------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{args.epochs}] Train (LR: {current_lr:.1e})", leave=True)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(train_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * args.accum_steps * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels).item()
            train_total += images.size(0)

            train_bar.set_postfix({
                'loss': f"{loss.item() * args.accum_steps:.4f}",
                'acc': f"{(train_correct / train_total) * 100:.2f}%"
            })

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # ---------------- FASE VALIDASI ----------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1:02d}/{args.epochs}] Val  ", leave=False)

        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = eval_criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += images.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        elapsed_sec = time.time() - start_time

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)

        print(f"📊 [Epoch {epoch+1:02d}/{args.epochs} - {elapsed_sec/60:.2f} m] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | Val Macro F1: {epoch_val_f1:.4f}")

        # Simpan checkpoint terbaik berdasarkan Macro F1 pada Data Validasi
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            best_val_acc = epoch_val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, args.output_path)
            print(f"   🌟 CHECKPOINT TERBAIK DISIMPAN -> '{args.output_path}' (Val Macro F1: {best_val_f1:.4f} | Acc: {best_val_acc*100:.2f}%)")

        print("-" * 75)

    # Simpan riwayat pelatihan ke JSON
    history_path = args.output_path.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"💾 Riwayat pelatihan berhasil diekspor ke: {history_path}")

    # ==========================================================================
    # 4. EVALUASI AKHIR PADA DATA UJI MURNI (HELD-OUT TEST SET 10%)
    # ==========================================================================
    print("\n" + "=" * 75)
    print("🏆 EVALUASI AKHIR PADA HELD-OUT TEST SET (10% Data Belum Pernah Dilihat)")
    print("=" * 75)
    model.load_state_dict(best_weights)
    model.eval()

    test_preds = []
    test_probs = []
    test_targets = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing Model"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())  # Probabilitas kelas Real Image (1)
            test_targets.extend(labels.numpy())

    test_targets = np.array(test_targets)
    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)

    test_acc = (test_preds == test_targets).mean() * 100.0
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    test_auc = roc_auc_score(test_targets, test_probs) * 100.0

    print(f"\n🎯 Akurasi Data Uji (Test Accuracy) : {test_acc:.2f}%")
    print(f"🎯 Macro F1-Score Data Uji          : {test_f1:.4f}")
    print(f"🎯 ROC-AUC Score Data Uji           : {test_auc:.2f}%")
    print("\n📋 CLASSIFICATION REPORT (AIForensicNet V4 pada Data Test Murni):")
    print(classification_report(test_targets, test_preds, target_names=["AI-Generated (0)", "Real Image (1)"], digits=4))

    cm = confusion_matrix(test_targets, test_preds)
    print("📊 CONFUSION MATRIX:")
    print(f"  [[TN={cm[0,0]} (Benar AI),  FP={cm[0,1]} (Salah Real)],")
    print(f"   [FN={cm[1,0]} (Salah AI),   TP={cm[1,1]} (Benar Real)]]")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Teroptimasi AIForensicNet V4")
    parser.add_argument("--dataset-name", type=str, default="Thunderbeast/AI-Generated-vs-Real-Images-Datasets")
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default="best_aiforensicnet_v4_ai_vs_real.pth")

    args = parser.parse_args()
    train_model(args)
