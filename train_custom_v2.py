import os
import sys
import time
import copy
import argparse
import warnings
warnings.filterwarnings('ignore', message='.*Palette images with Transparency.*')
warnings.filterwarnings('ignore', message='.*xFormers is not available.*')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from datasets import load_dataset

from model_custom import AIForensicNetV2, AIForensicNet


# ==============================================================================
# DATASET & AUGMENTASI KHUSUS FORENSIK
# ==============================================================================

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


def get_forensic_transforms(img_size=224):
    """
    Augmentasi data yang dirancang khusus untuk forensik citra AI:
    - Menjaga integritas sinyal artefak mikro (tidak menggunakan kompresi JPEG berat atau blur agresif).
    - Menambahkan variasi flip, rotasi halus, dan skala kecil untuk mencegah overfitting.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.90, 1.0), ratio=(0.95, 1.05)),
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
# LOSS DENGAN KNOWLEDGE DISTILLATION (OPSIONAL TEACHER GUIDANCE)
# ==============================================================================

class DistillationLoss(nn.Module):
    """
    Loss gabungan: Label-Smoothed Hard Cross-Entropy + Soft KL-Divergence dari Teacher.
    Memungkinkan AIForensicNetV2 (Student) menyerap representasi fitur mendalam
    dari Teacher (misal ConvNeXt-Tiny) yang sudah memiliki akurasi >95.8%.
    """
    def __init__(self, teacher_model, alpha=0.5, temperature=3.0, label_smoothing=0.05):
        super().__init__()
        self.teacher = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl = nn.KLDivLoss(reduction="batchmean")

        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def forward(self, student_logits, labels, images):
        hard_loss = self.ce(student_logits, labels)
        if self.teacher is None:
            return hard_loss

        with torch.no_grad():
            teacher_logits = self.teacher(images)

        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl(soft_student, soft_teacher) * (self.temperature ** 2)

        return (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss


def build_teacher(teacher_name, checkpoint_path, device):
    """Membangun dan memuat bobot model teacher untuk knowledge distillation."""
    print(f"👨‍🏫 Memuat Teacher Model: {teacher_name} dari {checkpoint_path}")
    if "convnext" in teacher_name.lower():
        m = models.convnext_tiny(weights=None)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_f, 2)
        )
    else:
        raise ValueError(f"Teacher architecture '{teacher_name}' belum didukung.")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    m.load_state_dict(state_dict)
    m = m.to(device)
    m.eval()
    return m


# ==============================================================================
# MAIN TRAINING WORKFLOW
# ==============================================================================

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("🚀 PELATIHAN TEROPTIMASI AIForensicNetV2 (AI vs Real Detection)")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"Device        : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device        : CPU (Peringatan: Pelatihan akan lambat tanpa GPU)")
    print(f"Total Epochs  : {args.epochs}")
    print(f"Batch Size    : {args.batch_size} (Akumulasi: {args.accum_steps} -> Efektif: {args.batch_size * args.accum_steps})")
    print(f"Learning Rate : {args.lr} (Warmup 2 Ep + Cosine Annealing)")
    print(f"Label Smooth  : {args.label_smoothing}")
    print(f"Distilasi KD  : {'Aktif' if args.distill else 'Nonaktif'}")
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

    train_transforms, val_transforms = get_forensic_transforms()
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

    # 2. Inisialisasi Model AIForensicNetV2
    model = AIForensicNetV2(num_classes=2, drop_rate=args.drop_rate, drop_path_rate=args.drop_path).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Arsitektur AIForensicNetV2 Terinisialisasi:")
    print(f"Total Parameter: {total_params:,} (~{total_params/1e6:.2f}M) - From Scratch")

    # 3. Setup Teacher jika Distilasi diaktifkan
    teacher_model = None
    if args.distill:
        if not os.path.exists(args.teacher_path):
            print(f"⚠️ Checkpoint teacher '{args.teacher_path}' tidak ditemukan. Menonaktifkan distilasi.")
        else:
            teacher_model = build_teacher("ConvNeXt-Tiny", args.teacher_path, device)

    criterion = DistillationLoss(
        teacher_model=teacher_model,
        alpha=0.5,
        temperature=3.0,
        label_smoothing=args.label_smoothing
    )
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Schedulers: 2 epoch linear warmup dilanjutkan Cosine Annealing
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
                loss = criterion(outputs, labels, images)
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
        all_val_targets = []

        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Validasi", leave=False)
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
                all_val_targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_targets, all_val_preds, average='macro')
        elapsed = time.time() - start_time

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)

        print(f"\n📊 [Epoch {epoch+1}/{args.epochs} - {elapsed:.1f}s] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | Val F1: {epoch_val_f1*100:.2f}%")

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            best_val_acc = epoch_val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, args.output_path)
            print(f"⭐ Checkpoint model terbaik disimpan ke '{args.output_path}' (Val F1: {best_val_f1*100:.2f}%)")

    # 4. Evaluasi Final Menggunakan Model Terbaik
    print("\n" + "=" * 70)
    print("📈 EVALUASI AKHIR PADA DATASET VALIDASI DENGAN CHECKPOINT TERBAIK")
    print("=" * 70)
    model.load_state_dict(best_weights)
    model.eval()

    # Evaluasi Standard vs TTA
    preds_standard = []
    preds_tta = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluasi Standard vs TTA"):
            images = images.to(device)
            images_flip = torch.flip(images, dims=[3])

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out_orig = model(images)
                out_flip = model(images_flip)

            probs_orig = torch.softmax(out_orig, dim=1)
            probs_flip = torch.softmax(out_flip, dim=1)
            probs_tta = (probs_orig + probs_flip) / 2.0

            _, p_std = torch.max(probs_orig, 1)
            _, p_tta = torch.max(probs_tta, 1)

            preds_standard.extend(p_std.cpu().numpy())
            preds_tta.extend(p_tta.cpu().numpy())
            targets.extend(labels.numpy())

    targets = np.array(targets)
    preds_standard = np.array(preds_standard)
    preds_tta = np.array(preds_tta)

    acc_std = np.mean(preds_standard == targets) * 100.0
    acc_tta = np.mean(preds_tta == targets) * 100.0

    print(f"\n🎯 Akurasi Validasi Standar : {acc_std:.2f}%")
    print(f"🚀 Akurasi Validasi TTA     : {acc_tta:.2f}% (Peningkatan: {acc_tta - acc_std:+.2f}%)")

    print("\n📋 Classification Report (TTA Mode):")
    print(classification_report(targets, preds_tta, target_names=["AI-Generated (0)", "Real Image (1)"], digits=4))

    print("\n✅ Pelatihan & Evaluasi Selesai! Model siap digunakan di app.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Teroptimasi AIForensicNetV2")
    parser.add_argument("--epochs", type=int, default=15, help="Jumlah total epoch (Disarankan 15-20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch per iterasi")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps (efektif batch = batch_size * accum_steps)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate puncak")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Nilai label smoothing")
    parser.add_argument("--drop-rate", type=float, default=0.2, help="Dropout probability pada classification head")
    parser.add_argument("--drop-path", type=float, default=0.1, help="Max drop path rate untuk Stochastic Depth")
    parser.add_argument("--distill", action="store_true", help="Gunakan knowledge distillation dari ConvNeXt-Tiny teacher")
    parser.add_argument("--teacher-path", type=str, default="best_convnext_tiny_ai_vs_real.pth", help="Path bobot model teacher")
    parser.add_argument("--output-path", type=str, default="best_aiforensicnet_v2_ai_vs_real.pth", help="Path output bobot terbaik")
    parser.add_argument("--dataset-name", type=str, default="Thunderbeast/AI-Generated-vs-Real-Images-Datasets", help="Nama dataset Hugging Face")
    parser.add_argument("--cache-dir", type=str, default="./data", help="Direktori cache dataset")
    parser.add_argument("--num-workers", type=int, default=0, help="Jumlah worker DataLoader")

    args = parser.parse_args()
    train_model(args)
