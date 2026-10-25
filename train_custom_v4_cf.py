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
from datasets import load_dataset, ClassLabel

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


class RandomDownscaleJitter:
    """
    Simulasi downsampling acak (skala 0.6x - 1.0x) dengan interpolasi bervariasi.
    Melatih model agar mengenali bahwa citra nyata yang sudah di-resize/dikompresi
    oleh aplikasi perpesanan (WhatsApp, Telegram) tetaplah Real Image dan tidak salah
    dideteksi sebagai AI akibat hilangnya sebagian noise sensor kamera.
    """
    def __init__(self, p=0.35, scale_range=(0.6, 1.0)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, img):
        if random.random() < self.p:
            w, h = img.size
            factor = random.uniform(*self.scale_range)
            nw = max(224, int(round(w * factor)))
            nh = max(224, int(round(h * factor)))
            resample_mode = random.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC])
            return img.resize((nw, nh), resample=resample_mode)
        return img


class NativeResolutionCrop:
    """
    Penyempurnaan Fundamental V4:
    Mempertahankan resolusi murni citra (tanpa bilinear downsampling global yang
    meredam sinyal frekuensi tinggi). Citra di-crop secara acak pada ukuran
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


class CommunityForensicsDataset(Dataset):
    """
    Adapter PyTorch Dataset untuk OwensLab/CommunityForensics.
    
    Karakteristik Khusus Dataset CommunityForensics:
    1. Kolom citra berupa 'image_data' bertipe byte mentah (raw bytes), di-decode via BytesIO.
    2. Penyelarasan Label (Label Re-Alignment):
       - Asli CommunityForensics: 1 = Fake (AI-Generated), 0 = Real Image.
       - Standar Model Proyek kita: 0 = AI-Generated, 1 = Real Image.
       - Dikonversi via: target_label = 0 if raw_label == 1 else 1.
    """
    def __init__(self, hf_data, transform=None):
        self.data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Decode citra dari bytes atau PIL jika sudah di-decode
        if "image_data" in item and item["image_data"] is not None:
            raw_bytes = item["image_data"]
            image = Image.open(BytesIO(raw_bytes))
        elif "image" in item and item["image"] is not None:
            image = item["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image))
        else:
            raise KeyError("Tidak ditemukan kolom 'image_data' atau 'image' pada sample dataset.")

        # Standarisasi format warna
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Penyelarasan label: CommunityForensics (1: Fake, 0: Real) -> Model (0: AI, 1: Real)
        raw_label = int(item["label"])
        target_label = 0 if raw_label == 1 else 1

        if self.transform:
            image = self.transform(image)

        return image, target_label


def get_v4_transforms(img_size=224):
    """
    Pipeline Transformasi AIForensicNet V4:
    - Kompresi JPEG acak pada resolusi asli.
    - Native Resolution Random Crop (menjaga artefak grid generator tetap utuh).
    - Random Horizontal Flip (tanpa rotasi bersudut hitam).
    - Gaussian blur mikro dan Color Jitter sangat halus.
    """
    train_transforms = transforms.Compose([
        RandomDownscaleJitter(p=0.35, scale_range=(0.6, 1.0)),
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
# 3. WORKFLOW PELATIHAN DAN EVALUASI V4 DENGAN COMMUNITYFORENSICS
# ==============================================================================

def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("🚀 PELATIHAN AIForensicNet V4 (Tri-Stream NPR + Spectral FFT + Native Crop)")
    print("📚 Sumber Data : OwensLab/CommunityForensics (CVPR 2025)")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"Device Utama        : CUDA ({torch.cuda.get_device_name(0)})")
        print(f"Total Alokasi VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"Device Utama        : CPU")
    print(f"Split Dataset       : {args.split}")
    print(f"Maksimal Sampel     : {'Semua' if args.max_samples <= 0 else f'{args.max_samples:,} citra'}")
    print(f"Total Epoch         : {args.epochs}")
    print(f"Batch Size          : {args.batch_size} (Akumulasi: {args.accum_steps} -> Efektif: {args.batch_size * args.accum_steps})")
    print(f"Learning Rate Puncak: {args.lr} (Warmup 2 Ep + Cosine Annealing)")
    print(f"Label Smoothing     : {args.label_smoothing}")
    print(f"Model Checkpoint    : {args.output_path}")
    print("=" * 80)

    # 1. Pemuatan Dataset dari OwensLab/CommunityForensics
    # Menggunakan data_files secara eksplisit agar TIDAK PERNAH mengunduh split Systematic (726 GB)
    if args.data_files:
        target_files = [f.strip() for f in args.data_files.split(",")]
    elif args.split == "PublicEval":
        if args.num_shards > 0:
            target_files = [f"data/publicEval/HFCF_PublicEval_{i}.parquet" for i in range(min(args.num_shards, 6))]
        else:
            target_files = ["data/publicEval/*.parquet"]
    else:
        target_files = None

    if target_files:
        print(f"\n📦 Memuat {len(target_files)} parquet file dari '{args.dataset_name}' (Target Disk Aman):")
        for tf in target_files:
            print(f"  ├─ {tf}")
        raw_dataset = load_dataset(
            args.dataset_name,
            data_files=target_files,
            split="train",
            cache_dir=args.cache_dir,
            verification_mode="no_checks"
        )
    else:
        print(f"\n📦 Memuat dataset '{args.dataset_name}' (split: '{args.split}') ...")
        raw_dataset = load_dataset(
            args.dataset_name,
            split=args.split,
            cache_dir=args.cache_dir,
            verification_mode="no_checks"
        )

    total_loaded = len(raw_dataset)
    print(f"Berhasil memuat dataset dengan total {total_loaded:,} citra.")

    # Pastikan kolom label bertipe ClassLabel agar mendukung stratify_by_column di HuggingFace datasets
    if not isinstance(raw_dataset.features["label"], ClassLabel):
        raw_dataset = raw_dataset.cast_column("label", ClassLabel(num_classes=2, names=["0", "1"]))

    # Opsional subsampling terstratifikasi untuk efisiensi di Google Colab
    if args.max_samples > 0 and args.max_samples < total_loaded:
        print(f"⚡ Melakukan sampling terstratifikasi sebanyak {args.max_samples:,} citra untuk efisiensi Colab...")
        sub_split = raw_dataset.train_test_split(
            train_size=args.max_samples,
            stratify_by_column="label",
            seed=args.seed
        )
        working_dataset = sub_split["train"]
    else:
        working_dataset = raw_dataset

    # Analisis distribusi label dalam subset kerja
    raw_labels = working_dataset["label"]
    fake_count = sum(1 for l in raw_labels if l == 1)
    real_count = sum(1 for l in raw_labels if l == 0)
    print(f"├─ Total Sampel Digunakan  : {len(working_dataset):,} citra")
    print(f"├─ Sampel AI-Generated (1) : {fake_count:,} ({fake_count/len(working_dataset)*100:.1f}%) -> Dipetakan ke Label 0")
    print(f"└─ Sampel Real Image (0)   : {real_count:,} ({real_count/len(working_dataset)*100:.1f}%) -> Dipetakan ke Label 1")

    # 2. Pembagian Dataset 3-Way Berstrata (80% Train, 10% Val, 10% Test)
    split1 = working_dataset.train_test_split(test_size=0.20, stratify_by_column="label", seed=args.seed)
    train_hf = split1["train"]
    val_test_hf = split1["test"]

    split2 = val_test_hf.train_test_split(test_size=0.50, stratify_by_column="label", seed=args.seed)
    val_hf = split2["train"]
    test_hf = split2["test"]

    print("\n📊 Rincian Pembagian 3-Way Stratified Data:")
    print(f"├─ Data Latih (Train 80%)    : {len(train_hf):,} citra")
    print(f"├─ Data Validasi (Val 10%)   : {len(val_hf):,} citra (digunakan untuk checkpointing)")
    print(f"└─ Data Uji Murni (Test 10%) : {len(test_hf):,} citra (disentuh 1x saja di akhir)")

    train_transforms, eval_transforms = get_v4_transforms(img_size=args.img_size)
    train_ds = CommunityForensicsDataset(train_hf, transform=train_transforms)
    val_ds = CommunityForensicsDataset(val_hf, transform=eval_transforms)
    test_ds = CommunityForensicsDataset(test_hf, transform=eval_transforms)

    use_pin = torch.cuda.is_available()
    use_persistent = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=use_pin, persistent_workers=use_persistent)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin, persistent_workers=use_persistent)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin, persistent_workers=use_persistent)

    # 3. Inisialisasi Model AIForensicNetV4
    model = AIForensicNetV4(num_classes=2, dropout_p=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🧠 Arsitektur AIForensicNet V4 Berhasil Diinisialisasi:")
    print(f"├─ Total Parameter    : {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"├─ Trainable Parameter: {trainable_params:,} (~{trainable_params/1e6:.2f}M)")
    print(f"└─ Fitur Forensik     : Tri-Stream (15-ch Multi-Scale + NPR, 2D-FFT Spectral, Spatial RGB) + MultiStatPool2d")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = min(2, max(1, args.epochs // 4))
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

    # 4. Loop Pelatihan & Validasi Tiap Epoch
    print(f"\n🚀 Memulai Pelatihan ({args.epochs} Epochs)...")
    train_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        optimizer.zero_grad(set_to_none=True)

        train_bar = tqdm(train_loader, desc=f"Ep [{epoch+1:02d}/{args.epochs:02d}] Train", leave=False)
        for step, (images, labels) in enumerate(train_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels) / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * args.accum_steps * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += images.size(0)

            train_bar.set_postfix({
                "loss": f"{loss.item() * args.accum_steps:.4f}",
                "acc": f"{correct_train / total_train * 100:.2f}%"
            })

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train * 100.0

        # Evaluasi Validasi
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = eval_criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)

                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_hf)
        val_acc = (np.array(val_preds_list) == np.array(val_targets_list)).mean() * 100.0
        val_f1 = f1_score(val_targets_list, val_preds_list, average='macro')

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        saved_flag = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, args.output_path)
            saved_flag = "⭐ [BEST CHECKPOINT DISIMPAN]"

        print(
            f"Epoch [{epoch+1:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.4f} {saved_flag}"
        )

    total_training_time = (time.time() - train_start_time) / 60.0
    print(f"\nPelatihan tuntas dalam {total_training_time:.2f} menit!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% | Best Macro F1: {best_val_f1:.4f}")

    # 5. Evaluasi Akhir pada Himpunan Uji Murni (Test Set)
    print("\n" + "=" * 80)
    print("🔬 EVALUASI AKHIR PADA TEST SET MURNI (HELD-OUT TEST SET)")
    print("=" * 80)

    # Muat checkpoint terbaik
    model.load_state_dict(best_weights)
    model.eval()

    test_preds_list = []
    test_targets_list = []
    test_probs_list = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            test_preds_list.extend(preds.cpu().numpy())
            test_targets_list.extend(labels.cpu().numpy())
            test_probs_list.extend(probs[:, 1].cpu().numpy())

    test_preds = np.array(test_preds_list)
    test_targets = np.array(test_targets_list)
    test_probs = np.array(test_probs_list)

    test_acc = (test_preds == test_targets).mean() * 100.0
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    test_auc = roc_auc_score(test_targets, test_probs) * 100.0

    print(f"\n🎯 Akurasi Data Uji (Test Accuracy) : {test_acc:.2f}%")
    print(f"🎯 Macro F1-Score Data Uji          : {test_f1:.4f}")
    print(f"🎯 ROC-AUC Score Data Uji           : {test_auc:.2f}%")
    print("\n📋 CLASSIFICATION REPORT (AIForensicNet V4 pada CommunityForensics):")
    print(classification_report(test_targets, test_preds, target_names=["AI-Generated (0)", "Real Image (1)"], digits=4))

    cm = confusion_matrix(test_targets, test_preds)
    print("📊 CONFUSION MATRIX:")
    print(f"  [[TN={cm[0,0]} (Benar AI),  FP={cm[0,1]} (Salah Real)],")
    print(f"   [FN={cm[1,0]} (Salah AI),   TP={cm[1,1]} (Benar Real)]]")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Teroptimasi AIForensicNet V4 dengan OwensLab/CommunityForensics")
    parser.add_argument("--dataset-name", type=str, default="OwensLab/CommunityForensics", help="Nama dataset Hugging Face")
    parser.add_argument("--split", type=str, default="PublicEval", help="Split data (default: PublicEval yang berpasangan AI+Real)")
    parser.add_argument("--num-shards", type=int, default=2, help="Jumlah shard parquet PublicEval (1=~8.5k, 2=~17k, 6=51.8k full, default: 2 shard ~7.4GB disk)")
    parser.add_argument("--data-files", type=str, default=None, help="Kustom path parquet file spesifik dipisah koma")
    parser.add_argument("--max-samples", type=int, default=16000, help="Jumlah sampel maksimal terstratifikasi untuk efisiensi Colab (default: 16000, -1 untuk semua)")
    parser.add_argument("--cache-dir", type=str, default="./data", help="Direktori cache lokal")
    parser.add_argument("--epochs", type=int, default=15, help="Jumlah epoch pelatihan")
    parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch per iterasi")
    parser.add_argument("--accum-steps", type=int, default=1, help="Langkah akumulasi gradien")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate puncak")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="L2 weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing regularization")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability pada classifier head")
    parser.add_argument("--img-size", type=int, default=224, help="Ukuran crop citra forensik")
    parser.add_argument("--num-workers", type=int, default=0, help="Jumlah worker DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed reproduktifitas")
    parser.add_argument("--output-path", type=str, default="best_aiforensicnet_v4_cf.pth", help="Path penyimpanan bobot terbaik")

    args = parser.parse_args()
    train_model(args)
