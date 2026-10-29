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
from datasets import load_dataset, ClassLabel

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
        image = item.get("image", item.get("Image"))
        label = int(item["label"])

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

    # 1. Muat Dataset dengan Split Resmi Defactify (Train: 42,000 | Validation: 9,000 | Test: 45,000)
    print(f"\n📦 Memuat split resmi dataset '{args.dataset_name}' dari Hugging Face / cache...")

    def prepare_defactify_split(dataset_split):
        """
        Standarisasi format kolom dan mapping label untuk split dataset Defactify:
        - Normalisasi nama kolom citra ('Image' -> 'image')
        - Mapping label: Defactify Label_A (0=Real, 1=AI) -> Standar V3 (0=AI-Generated, 1=Real Image)
        """
        if "Label_A" in dataset_split.column_names:
            dataset_split = dataset_split.map(
                lambda b: {"label": [0 if int(x) == 1 else 1 for x in b["Label_A"]]},
                batched=True
            )
            dataset_split = dataset_split.cast_column("label", ClassLabel(num_classes=2, names=["AI-Generated", "Real Image"]))
        elif "label" in dataset_split.column_names and not isinstance(dataset_split.features.get("label"), ClassLabel):
            dataset_split = dataset_split.cast_column("label", ClassLabel(num_classes=2, names=["AI-Generated", "Real Image"]))

        if "Image" in dataset_split.column_names and "image" not in dataset_split.column_names:
            dataset_split = dataset_split.rename_column("Image", "image")

        return dataset_split

    try:
        from datasets import get_dataset_split_names
        available_splits = get_dataset_split_names(args.dataset_name)
    except Exception:
        available_splits = ["train", "validation", "test"]

    if "validation" in available_splits and "test" in available_splits:
        print("[*] Split resmi Defactify terdeteksi:")
        print("    ├─ 'train'      : 42,000 citra (Rekomendasi untuk Model Training)")
        print("    ├─ 'validation' :  9,000 citra (Rekomendasi untuk Hyperparameter Tuning & Early Stopping)")
        print("    └─ 'test'       : 45,000 citra (Rekomendasi untuk Final Unbiased Benchmark Evaluation)")

        train_hf = load_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir, verification_mode="no_checks")
        val_hf = load_dataset(args.dataset_name, split="validation", cache_dir=args.cache_dir, verification_mode="no_checks")
        test_hf = load_dataset(args.dataset_name, split="test", cache_dir=args.cache_dir, verification_mode="no_checks")

        train_hf = prepare_defactify_split(train_hf)
        val_hf = prepare_defactify_split(val_hf)
        test_hf = prepare_defactify_split(test_hf)

        # Opsi subsampling jika pengguna menetapkan kuota sampel
        if args.max_samples is not None and args.max_samples > 0:
            total_available = len(train_hf) + len(val_hf) + len(test_hf)
            if total_available > args.max_samples:
                # Subsampling proporsional sesuai rasio resmi Defactify (42k : 9k : 45k)
                ratio_train = len(train_hf) / total_available  # ~43.75%
                ratio_val = len(val_hf) / total_available      # ~9.375%
                n_train = max(10, int(args.max_samples * ratio_train))
                n_val = max(5, int(args.max_samples * ratio_val))
                n_test = max(10, args.max_samples - n_train - n_val)
                print(f"[*] Subsampling proporsional resmi ({args.max_samples:,} total): Train={n_train:,}, Val={n_val:,}, Test={n_test:,}...")
                train_hf = train_hf.shuffle(seed=42).select(range(min(n_train, len(train_hf))))
                val_hf = val_hf.shuffle(seed=42).select(range(min(n_val, len(val_hf))))
                test_hf = test_hf.shuffle(seed=42).select(range(min(n_test, len(test_hf))))
        elif any([args.max_train_samples, args.max_val_samples, args.max_test_samples]):
            if args.max_train_samples and len(train_hf) > args.max_train_samples:
                print(f"[*] Subsampling train: {args.max_train_samples:,} citra...")
                train_hf = train_hf.shuffle(seed=42).select(range(args.max_train_samples))
            if args.max_val_samples and len(val_hf) > args.max_val_samples:
                print(f"[*] Subsampling val: {args.max_val_samples:,} citra...")
                val_hf = val_hf.shuffle(seed=42).select(range(args.max_val_samples))
            if args.max_test_samples and len(test_hf) > args.max_test_samples:
                print(f"[*] Subsampling test: {args.max_test_samples:,} citra...")
                test_hf = test_hf.shuffle(seed=42).select(range(args.max_test_samples))
    else:
        # Fallback jika repositori kustom hanya menyediakan single split 'train'
        raw_dataset = load_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir, verification_mode="no_checks")
        raw_dataset = prepare_defactify_split(raw_dataset)
        if args.max_samples is not None and args.max_samples > 0 and len(raw_dataset) > args.max_samples:
            raw_dataset = raw_dataset.shuffle(seed=42).select(range(args.max_samples))
        train_split = raw_dataset.train_test_split(test_size=0.20, stratify_by_column="label", seed=42)
        train_hf = train_split["train"]
        temp_split = train_split["test"].train_test_split(test_size=0.50, stratify_by_column="label", seed=42)
        val_hf = temp_split["train"]
        test_hf = temp_split["test"]

    print("=" * 70)
    print("📊 DATASET DEFACTIFY TERKONFIGURASI (SPLIT RESMI):")
    print(f"├─ Data Train (Resmi)     : {len(train_hf):,} citra (Model Training)")
    print(f"├─ Data Validasi (Resmi)  : {len(val_hf):,} citra (Hyperparameter Tuning & Best Checkpoint)")
    print(f"└─ Data Pengujian (Resmi) : {len(test_hf):,} citra (Independent Benchmark Evaluation)")
    print("=" * 70)


    train_transforms, val_transforms = get_robust_forensic_transforms()
    train_ds = HFToPyTorchDataset(train_hf, transform=train_transforms)
    val_ds = HFToPyTorchDataset(val_hf, transform=val_transforms)
    test_ds = HFToPyTorchDataset(test_hf, transform=val_transforms)

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
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

    # Learning Rate Scheduler: Warmup 3 Epoch + Cosine Annealing Terkalibrasi
    warmup_epochs = min(3, max(1, args.epochs // 4))
    cosine_epochs = max(1, args.epochs - warmup_epochs)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, cosine_epochs))
            return max(1e-3, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_score = 0.0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_probs = []
    best_val_labels = []
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
        epoch_val_probs = []
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
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)

                val_correct += torch.sum(preds == labels).item()
                val_total += images.size(0)

                epoch_val_probs.extend(probs[:, 1].cpu().numpy())  # Probabilitas kelas Real (1)
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

        # Checkpoint selection berdasarkan skor terbobot Macro F1 (70%) + Akurasi (30%)
        # Mencegah bias akurasi semu akibat dataset tidak seimbang (5:1)
        val_score = 0.70 * epoch_val_f1 + 0.30 * epoch_val_acc
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_acc = epoch_val_acc
            best_val_f1 = epoch_val_f1
            best_val_probs = copy.deepcopy(epoch_val_probs)
            best_val_labels = copy.deepcopy(all_val_labels)
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "val_acc": best_val_acc,
                "val_f1": best_val_f1,
                "epoch": epoch + 1
            }, args.output_path)
            print(f"   🌟 CHECKPOINT TERBAIK DIPERBARUI! (Val F1: {epoch_val_f1*100:.2f}%, Acc: {epoch_val_acc*100:.2f}%) -> '{args.output_path}'")

        print("-" * 70)

    # 3. Pencarian Threshold Keputusan Optimal dari Set Validasi
    print("\n🎯 Menghitung Threshold Kalibrasi Optimal pada Set Validasi...")
    best_threshold = 0.50
    best_th_f1 = best_val_f1
    if len(best_val_probs) > 0 and len(best_val_labels) > 0:
        for th in np.arange(0.30, 0.70, 0.02):
            th_preds = (np.array(best_val_probs) >= th).astype(int)
            th_f1 = f1_score(best_val_labels, th_preds, average='macro')
            if th_f1 > best_th_f1:
                best_th_f1 = th_f1
                best_threshold = float(th)

    print(f"   Threshold Standar     : 0.50 (Macro F1: {best_val_f1*100:.2f}%)")
    print(f"   Threshold Terkalibrasi : {best_threshold:.2f} (Macro F1: {best_th_f1*100:.2f}%)")

    # 4. Evaluasi Akhir pada Independent Test Set Resmi Defactify dengan TTA
    print(f"\n🔍 Memuat bobot terbaik untuk Evaluasi Final pada Test Set Resmi Defactify ({len(test_ds):,} citra)...")
    model.load_state_dict(best_weights)
    model.eval()

    probs_real_tta = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Test Evaluation (TTA)"):
            images = images.to(device, non_blocking=True)
            images_flip = torch.flip(images, dims=[3])

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out_std = model(images)
                out_flip = model(images_flip)

            prob_std = torch.softmax(out_std, dim=1)
            prob_flip = torch.softmax(out_flip, dim=1)
            prob_tta = (prob_std + prob_flip) / 2.0

            probs_real_tta.extend(prob_tta[:, 1].cpu().numpy())
            targets.extend(labels.numpy())

    probs_real_tta = np.array(probs_real_tta)
    targets = np.array(targets)

    # Prediksi Standar (Threshold 0.50)
    preds_std = (probs_real_tta >= 0.50).astype(int)
    acc_std = (preds_std == targets).mean() * 100.0
    f1_std = f1_score(targets, preds_std, average='macro') * 100.0

    # Prediksi Terkalibrasi (Threshold Optimal)
    preds_cal = (probs_real_tta >= best_threshold).astype(int)
    acc_cal = (preds_cal == targets).mean() * 100.0
    f1_cal = f1_score(targets, preds_cal, average='macro') * 100.0

    print("\n" + "=" * 70)
    print("🏆 HASIL AKHIR PELATIHAN AIForensicNet V3 (Official Defactify Test Set):")
    print(f"├─ Evaluasi Standar (Threshold 0.50)   : Akurasi = {acc_std:.2f}%, Macro F1 = {f1_std:.2f}%")
    print(f"└─ Evaluasi Terkalibrasi (Th {best_threshold:.2f}) : Akurasi = {acc_cal:.2f}%, Macro F1 = {f1_cal:.2f}% (Peningkatan: {f1_cal - f1_std:+.2f}%)")
    print("=" * 70)

    print("\nCLASSIFICATION REPORT (AIForensicNet V3 - Standard Threshold 0.50):")
    print(classification_report(targets, preds_std, target_names=["AI-Generated (0)", "Real Image (1)"], digits=4))

    print("\nCLASSIFICATION REPORT (AIForensicNet V3 - Calibrated Optimal Threshold):")
    print(classification_report(targets, preds_cal, target_names=["AI-Generated (0)", "Real Image (1)"], digits=4))

    # Perbarui file checkpoint dengan metadata threshold
    torch.save({
        "model_state_dict": best_weights,
        "val_acc": best_val_acc,
        "val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "test_acc_std": acc_std,
        "test_acc_cal": acc_cal
    }, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Teroptimasi AIForensicNet V3 (Defactify Official Splits)")
    parser.add_argument("--dataset-name", type=str, default="Rajarshi-Roy-research/Defactify_Image_Dataset", help="HuggingFace dataset repository (default: Rajarshi-Roy-research/Defactify_Image_Dataset)")
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--max-samples", type=int, default=None, help="Batas maksimal total sampel (dibagi proporsional ke train/val/test resmi Defactify)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Batas maksimal sampel train resmi Defactify (opsional)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Batas maksimal sampel validation resmi Defactify (opsional)")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Batas maksimal sampel test resmi Defactify (opsional)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate awal (default: 3e-4 untuk stabilitas maksimal)")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-path", type=str, default="best_aiforensicnet_v3_ai_vs_real.pth")

    args = parser.parse_args()
    train_model(args)


