import os
import sys
import time
import copy
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
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from datasets import load_dataset

from model_custom import AIForensicNetV3


# ==============================================================================
# 1. ANTI-COMPRESSION & ROBUST FORENSIC AUGMENTATION PIPELINE
# ==============================================================================

class RandomJPEGCompression:
    """
    Simulasi kompresi JPEG acak.
    Melatih model agar kebal terhadap degradasi kompresi media sosial (WhatsApp,
    Instagram, WebP, dsb.) dan tidak salah mengira artefak kompresi sebagai artefak AI.
    """
    def __init__(self, quality_range=(60, 95), p=0.45):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img


class HFToPyTorchDataset(Dataset):
    """Adapter menghubungkan Hugging Face Dataset dengan PyTorch DataLoader."""
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


def get_robust_forensic_transforms(img_size=224):
    """
    Pipeline augmentasi anti-kompresi & forensik tangguh:
    - Random JPEG Compression untuk kekebalan distorsi transmisi web.
    - Random Gaussian Blur mikro (mencegah overfitting pada piksel rapuh).
    - Color Jitter sangat halus (variasi lighting/kamera).
    - Random Horizontal Flip & Rotation halus.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        RandomJPEGCompression(quality_range=(60, 95), p=0.45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.20),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.92, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms


# ==============================================================================
# 2. MAIN TRAINING WORKFLOW AIFORENSICNET V3
# ==============================================================================

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("🚀 PELATIHAN AIForensicNet V3 (Multi-Scale Static Bank + Anti-Compression)")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"Device        : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device        : CPU (Peringatan: Pelatihan akan lambat tanpa GPU)")
    print(f"Total Epochs  : {args.epochs}")
    print(f"Batch Size    : {args.batch_size} (Akumulasi: {args.accum_steps} -> Efektif: {args.batch_size * args.accum_steps})")
    print(f"Learning Rate : {args.lr} (Warmup 2 Ep + Cosine Annealing)")
    print(f"Label Smooth  : {args.label_smoothing}")
    print(f"Output Path   : {args.output_path}")
    print("=" * 70)

    # 1. Muat Dataset
    print("\n📦 Memuat dataset dari cache...")
    raw_dataset = load_dataset(
        args.dataset_name,
        split="train",
        cache_dir=args.cache_dir,
        verification_mode="no_checks"
    )
    splits = raw_dataset.train_test_split(test_size=0.10, stratify_by_column="label", seed=42)
    train_hf = splits["train"]
    val_hf = splits["test"]

    print(f"Dataset Loaded : {len(raw_dataset):,} citra")
    print(f"├─ Data Train  : {len(train_hf):,} citra (90%)")
    print(f"└─ Data Validasi: {len(val_hf):,} citra (10%)")

    train_transforms, val_transforms = get_robust_forensic_transforms()
    train_ds = HFToPyTorchDataset(train_hf, transform=train_transforms)
    val_ds = HFToPyTorchDataset(val_hf, transform=val_transforms)

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_pin
    )

    # 2. Inisialisasi Model AIForensicNetV3
    model = AIForensicNetV3(num_classes=2, dropout_p=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🧠 Arsitektur AIForensicNetV3 Terinisialisasi:")
    print(f"├─ Total Parameter    : {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"└─ Trainable Parameter: {trainable_params:,} (~{trainable_params/1e6:.2f}M)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Learning Rate Scheduler: Warmup 2 epoch + Cosine Annealing
    warmup_epochs = min(2, args.epochs // 4)
    cosine_epochs = max(1, args.epochs - warmup_epochs)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, cosine_epochs))
            return max(1e-3, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    print("\n🏁 Memulai loop pelatihan...")
    for epoch in range(args.epochs):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # ---------------- FASE TRAINING ----------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Train (LR: {current_lr:.1e})", leave=True)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(train_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
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

        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Val  ", leave=True)

        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = eval_criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += images.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                val_bar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'val_acc': f"{(val_correct / val_total) * 100:.2f}%"
                })

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        elapsed_sec = time.time() - start_time

        print(f"📊 Ringkasan Epoch {epoch+1:02d}:")
        print(f"   Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}%")
        print(f"   Val Loss  : {epoch_val_loss:.4f} | Val Acc  : {epoch_val_acc * 100:.2f}% | Val Macro F1: {epoch_val_f1:.4f}")
        print(f"   Durasi    : {elapsed_sec / 60:.2f} menit")

        # Simpan checkpoint terbaik
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_f1 = epoch_val_f1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, args.output_path)
            print(f"   🌟 CHECKPOINT TERBAIK DIPERBARUI! Disimpan ke: '{args.output_path}'")

        print("-" * 70)

    # 3. Evaluasi Akhir dengan Test-Time Augmentation (TTA)
    print("\n🔍 Memuat bobot terbaik untuk Evaluasi Final...")
    model.load_state_dict(best_weights)
    model.eval()

    preds_std = []
    preds_tta = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Final TTA Evaluation"):
            images = images.to(device, non_blocking=True)
            images_flip = torch.flip(images, dims=[3])

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out_std = model(images)
                out_flip = model(images_flip)

            prob_std = torch.softmax(out_std, dim=1)
            prob_flip = torch.softmax(out_flip, dim=1)
            prob_tta = (prob_std + prob_flip) / 2.0

            _, p_s = torch.max(prob_std, 1)
            _, p_t = torch.max(prob_tta, 1)

            preds_std.extend(p_s.cpu().numpy())
            preds_tta.extend(p_t.cpu().numpy())
            targets.extend(labels.numpy())

    acc_std = (np.array(preds_std) == np.array(targets)).mean() * 100.0
    acc_tta = (np.array(preds_tta) == np.array(targets)).mean() * 100.0

    print("\n" + "=" * 70)
    print("🏆 HASIL AKHIR PELATIHAN AIForensicNet V3:")
    print(f"Akurasi Standar : {acc_std:.2f}%")
    print(f"Akurasi TTA     : {acc_tta:.2f}% (Peningkatan: {acc_tta - acc_std:+.2f}%)")
    print("=" * 70)
    print("\nCLASSIFICATION REPORT (AIForensicNet V3 + TTA):")
    print(classification_report(targets, preds_tta, target_names=["AI-Generated (0)", "Real Image (1)"], digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Teroptimasi AIForensicNet V3")
    parser.add_argument("--dataset-name", type=str, default="Thunderbeast/AI-Generated-vs-Real-Images-Datasets")
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-path", type=str, default="best_aiforensicnet_v3_ai_vs_real.pth")

    args = parser.parse_args()
    train_model(args)
