"""
Script Pelatihan AIForensicNet V4 pada Dataset HiDF (ACM SIGKDD 2025)
Mitigasi Berita Hoaks Menggunakan Deteksi Deepfake Berbasis Forensik Tri-Stream.
"""

import os
import time
import random
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model_custom import AIForensicNetV4
from hidf_utils import download_hidf_dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class RandomJPEGCompression:
    def __init__(self, quality_min=60, quality_max=95, p=0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(self.quality_min, self.quality_max)
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            return Image.open(buf).convert('RGB')
        return img


class HiDFDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['label']


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"Device Utama: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name    : {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 65)

    # 1. Unduh / Siapkan Dataset HiDF
    DATA_DIR = os.path.join("data", "hidf")
    N_SAMPLES_PER_CLASS = 1500
    real_dir, fake_dir = download_hidf_dataset(DATA_DIR, n_samples_per_class=N_SAMPLES_PER_CLASS, max_workers=8)

    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    data_list = [{'path': p, 'label': 0} for p in real_files] + [{'path': p, 'label': 1} for p in fake_files]
    df_dataset = pd.DataFrame(data_list)

    # 2. Stratified Split 80:10:10
    train_df, temp_df = train_test_split(df_dataset, test_size=0.20, stratify=df_dataset['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

    print(f"\n[+] Pembagian Data (80:10:10):")
    print(f"    - Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # 3. Transformasi & DataLoaders
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomJPEGCompression(quality_min=60, quality_max=95, p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HiDFDataset(train_df, train_transforms)
    val_dataset   = HiDFDataset(val_df,   eval_transforms)
    test_dataset  = HiDFDataset(test_df,  eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Inisialisasi Model AIForensicNet V4
    model = AIForensicNetV4(num_classes=2, dropout_p=0.25).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n[+] AIForensicNet V4 Diinisialisasi: {params:,} parameter (~{params/1e6:.2f}M)")

    # 5. Optimasi & Pelatihan
    EPOCHS = 15
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    BEST_MODEL_PATH = "best_aiforensicnet_v4_hidf.pth"
    best_val_f1 = 0.0

    print(f"\n[*] Memulai Pelatihan AIForensicNet V4 selama {EPOCHS} Epoch...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{EPOCHS}] Train", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100

        # Validasi
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_df)
        val_acc = (np.array(val_preds) == np.array(val_targets)).mean() * 100
        val_f1 = f1_score(val_targets, val_preds, average='macro') * 100

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                'epoch': int(epoch + 1),
                'model_state_dict': model.state_dict(),
                'val_acc': float(val_acc),
                'val_f1': float(val_f1)
            }, BEST_MODEL_PATH)
            status = f"★ BEST SAVED (F1: {best_val_f1:.2f}%)"
        else:
            status = ""

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.2f}% {status}")

    # 6. Evaluasi Test Set Unseen
    print("\n" + "=" * 65)
    print("EVALUASI AKHIR PADA TEST SET HIDF (10% UNSEEN):")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds = []
    test_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())

    print(classification_report(test_targets, test_preds, target_names=['Real', 'Fake (Deepfake)'], digits=4))
    print("=" * 65)


if __name__ == "__main__":
    main()
