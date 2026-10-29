"""
Pelatihan Mandiri AIFusionNet-Universal V5 (Revisi V5.1)
========================================================
Melatih model dari awal (from scratch, tanpa pretrained weights) pada multi-dataset:
1. CASIA v2.0 (Splicing & Copy-Move Image Tampering)
2. Defactify (COCO Real vs Modern AI Generators: Midjourney, SDXL, DALL-E)
3. Thunderbeast (Authentic Art vs AI-Generated Art)
4. IMD2020 (In-the-wild realistic image tampering)

Penyempurnaan Utama V5.1:
- Crop Resolusi Asli: Mempertahankan artefak frekuensi tinggi (SRM & FFT) tanpa distorsi bilinear.
- Anti-Shortcut JPEG: Re-kompresi JPEG pada kedua kelas (mencegah model belajar jalan pintas Au=JPEG, Tp=TIF).
- Penyeimbangan Per-Sumber: Setiap dataset diseimbangkan 50:50 secara mandiri + split stratified (source, label).
- Opsi Holdout Lintas-Dataset (--holdout): Uji generalisasi murni tanpa membocorkan dataset ke tahap training.
- Kalibrasi Anti-False-Positive: Temperature Scaling + ambang keputusan terkalibrasi berbasis target FPR (misal <= 5%).
- Training Stabil: Linear warmup per-step, gradient clipping, early stopping, dan CE + label smoothing.
- Evaluasi Lengkap: In-distribution test set, holdout lintas-dataset, dan uji ketahanan kompresi JPEG Q=75.
"""

import os
import sys
import math
import glob
import random
import argparse
import io
import copy
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model_universal_v5 import AIFusionNetUniversal, build_v5_model

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ['Real', 'Fake/AI/Tampered']


# ==============================================================================
# 1. FORENSIC PREPROCESSING & DATA AUGMENTATION
# ==============================================================================

def safe_save_rgb(img_obj: Image.Image, save_path: str):
    """Simpan citra secara aman tanpa UserWarning palet transparansi."""
    if img_obj.mode in ('RGBA', 'LA') or (img_obj.mode == 'P' and 'transparency' in getattr(img_obj, 'info', {})):
        img_obj = img_obj.convert('RGBA')
    img_obj.convert('RGB').save(save_path, 'JPEG', quality=95)


class RandomJPEGCompression:
    """Re-kompresi JPEG acak pada KEDUA kelas untuk mencegah jalan pintas format file."""
    def __init__(self, quality_range=(60, 100), p=0.8):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(self.quality_range[0], self.quality_range[1])
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            return Image.open(buf).convert('RGB')
        return img


class FixedJPEG:
    """Kompresi JPEG kualitas tetap (digunakan untuk stress test ketahanan)."""
    def __init__(self, quality: int = 75):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class RandomGaussianBlur:
    def __init__(self, radius_range=(0.2, 1.2), p=0.2):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(self.radius_range[0], self.radius_range[1])
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class EnsureMinSize:
    """Hanya memperbesar citra jika lebih kecil dari ukuran crop (citra besar tidak di-resize)."""
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if min(w, h) >= self.size:
            return img
        scale = float(self.size) / float(min(w, h))
        new_w, new_h = max(self.size, int(round(w * scale))), max(self.size, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.Resampling.BICUBIC)


def get_v5_transforms(img_size: int = 224, is_train: bool = True, use_crop: bool = True, jpeg_p: float = 0.8):
    """Pipeline transformasi citra forensik V5.1 (Single Scale)."""
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if use_crop:
        geo = [EnsureMinSize(img_size), transforms.RandomCrop(img_size)] if is_train else [EnsureMinSize(img_size), transforms.CenterCrop(img_size)]
    else:
        geo = [transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR)]

    if is_train:
        return transforms.Compose(geo + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            RandomGaussianBlur(radius_range=(0.2, 1.2), p=0.2),
            RandomJPEGCompression(quality_range=(60, 100), p=jpeg_p),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            norm
        ])
    else:
        return transforms.Compose(geo + [
            transforms.ToTensor(),
            norm
        ])


def get_salient_patch_pil(pil_img: Image.Image, size: int = 224) -> Image.Image:
    """
    Salient Gradient Patch Selection:
    Memilih patch resolusi 1:1 dengan varians tekstur / kerapatan tepi tertinggi
    dari 5 kandidat (Center, 4 Sudut) agar tidak memotong area kosong/langit datar pada citra splicing.
    """
    w, h = pil_img.size
    if min(w, h) < size:
        scale = float(size) / float(min(w, h))
        pil_img = pil_img.resize((max(size, int(round(w * scale))), max(size, int(round(h * scale)))), Image.Resampling.BICUBIC)
        w, h = pil_img.size
    if w == size and h == size:
        return pil_img

    candidates = [
        ((w - size) // 2, (h - size) // 2),  # Center
        (0, 0),                              # Top-Left
        (w - size, 0),                      # Top-Right
        (0, h - size),                      # Bottom-Left
        (w - size, h - size),               # Bottom-Right
    ]
    best_patch = None
    best_score = -1.0
    for left, top in candidates:
        crop = pil_img.crop((left, top, left + size, top + size))
        crop_gray = np.array(crop.convert('L'), dtype=np.float32)
        score = float(np.std(crop_gray))
        if score > best_score:
            best_score = score
            best_patch = crop
    return best_patch if best_patch is not None else pil_img.crop((0, 0, size, size))


def get_dual_scale_transforms(img_size: int = 224, is_train: bool = True, jpeg_p: float = 0.8):
    """Pipeline transformasi citra forensik Dual-Scale V5.3 (Salient Patch + Rotation)."""
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # 1. Global View (Resized 224x224 untuk Makro Semantik)
    if is_train:
        global_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            norm
        ])
    else:
        global_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            norm
        ])

    # 2. Native Micro-Forensic Patch (1:1 Resolution Tanpa Downsampling untuk SRM & FFT)
    if is_train:
        patch_tf = transforms.Compose([
            EnsureMinSize(img_size),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            RandomGaussianBlur(radius_range=(0.2, 1.2), p=0.2),
            RandomJPEGCompression(quality_range=(60, 100), p=jpeg_p),
            transforms.ToTensor(),
            norm
        ])
    else:
        patch_tf = transforms.Compose([
            transforms.ToTensor(),
            norm
        ])

    return global_tf, patch_tf


# ==============================================================================
# 2. DATASET DEFINITION & LOADER
# ==============================================================================

class UniversalForensicDataset(Dataset):
    """PyTorch Dataset untuk citra forensik multi-domain (Single-Scale)."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['label'])
        source = row.get('source', 'Unknown')

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        return img_tensor, label, source


class DualScaleForensicDataset(Dataset):
    """
    PyTorch Dataset untuk citra forensik Dual-Scale:
    - Skala 1 (Global): Citra utuh di-resize ke 224x224 untuk analisis konteks semantik makro.
    - Skala 2 (Patch): Crop 224x224 pada resolusi murni 1:1 tanpa resize untuk analisis residu mikro SRM & 2D FFT.
    """
    def __init__(self, df: pd.DataFrame, global_tf=None, patch_tf=None, is_eval: bool = False):
        self.df = df.reset_index(drop=True)
        self.global_tf = global_tf
        self.patch_tf = patch_tf
        self.is_eval = is_eval

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['label'])
        source = row.get('source', 'Unknown')

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        g_img = self.global_tf(img) if self.global_tf is not None else transforms.ToTensor()(img)
        if self.is_eval:
            p_pil = get_salient_patch_pil(img, 224)
            p_img = self.patch_tf(p_pil) if self.patch_tf is not None else transforms.ToTensor()(p_pil)
        else:
            p_img = self.patch_tf(img) if self.patch_tf is not None else g_img

        return (g_img, p_img), label, source


# ==============================================================================
# 3. DATASET EXTRACTORS
# ==============================================================================

def load_casia2_dataset(casia_dir: str = "data/casia2", n_samples_per_class: int = None, seed: int = 42) -> List[Dict]:
    """Pemuat CASIA v2.0 (Au = Authentic Real, Tp = Tampered Splicing/Copy-Move)."""
    records = []
    search_dirs = [casia_dir, os.path.expanduser("~/.cache/kagglehub")]
    au_files, tp_files = [], []
    for d in search_dirs:
        if os.path.exists(d):
            au_files.extend([p for p in glob.glob(f"{d}/**/Au/*", recursive=True) if p.lower().endswith(VALID_EXTS)])
            tp_files.extend([p for p in glob.glob(f"{d}/**/Tp/*", recursive=True) if p.lower().endswith(VALID_EXTS)])

    if not au_files or not tp_files:
        try:
            import kagglehub
            print("[*] Mengunduh dataset CASIA v2.0 via KaggleHub CDN...")
            downloaded_dir = kagglehub.dataset_download('divg07/casia-20-image-tampering-detection-dataset')
            au_files = [p for p in glob.glob(f"{downloaded_dir}/**/Au/*", recursive=True) if p.lower().endswith(VALID_EXTS)]
            tp_files = [p for p in glob.glob(f"{downloaded_dir}/**/Tp/*", recursive=True) if p.lower().endswith(VALID_EXTS)]
        except Exception:
            pass

    au_files = sorted(list(set(au_files)))
    tp_files = sorted(list(set(tp_files)))

    if not au_files or not tp_files:
        print(f"[!] Direktori CASIA v2.0 tidak ditemukan di {casia_dir}. Melewati...")
        return records

    if n_samples_per_class is not None and n_samples_per_class > 0:
        rng = random.Random(seed)
        if len(au_files) > n_samples_per_class:
            au_files = rng.sample(au_files, n_samples_per_class)
        if len(tp_files) > n_samples_per_class:
            tp_files = rng.sample(tp_files, n_samples_per_class)

    for p in au_files:
        records.append({'image_path': p, 'label': 0, 'source': 'CASIA2'})
    for p in tp_files:
        records.append({'image_path': p, 'label': 1, 'source': 'CASIA2'})

    print(f"[+] CASIA v2.0: {len(au_files):,} Real | {len(tp_files):,} Tampered.")
    return records


def count_images_in_dir(path: str) -> int:
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.lower().endswith(VALID_EXTS)])


def load_defactify_dataset(cache_dir: str = "data/defactify_cache", n_samples_per_class: int = 4000) -> List[Dict]:
    """Pemuat Defactify (COCO Real vs Modern AI Generated)."""
    real_dir = os.path.join(cache_dir, "real")
    ai_dir = os.path.join(cache_dir, "ai")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    c_real = count_images_in_dir(real_dir)
    c_ai = count_images_in_dir(ai_dir)
    target_cnt = float('inf') if (n_samples_per_class is None or n_samples_per_class <= 0) else n_samples_per_class

    if min(c_real, c_ai) < target_cnt:
        print(f"[*] Mengambil streaming Defactify dari Hugging Face...")
        try:
            if load_dataset is None:
                raise ImportError("Library 'datasets' belum terinstal. Jalankan: pip install datasets")
            ds = load_dataset('Rajarshi-Roy-research/Defactify_Image_Dataset', split='train', streaming=True)
            seen_r, seen_f = 0, 0
            for row in tqdm(ds, desc="Defactify"):
                if c_real >= target_cnt and c_ai >= target_cnt:
                    break
                lbl = int(row['Label_A'])
                if lbl == 0:
                    seen_r += 1
                    if seen_r <= c_real or c_real >= target_cnt:
                        continue
                    safe_save_rgb(row['Image'], os.path.join(real_dir, f"def_real_{c_real:05d}.jpg"))
                    c_real += 1
                elif lbl == 1:
                    seen_f += 1
                    if seen_f <= c_ai or c_ai >= target_cnt:
                        continue
                    safe_save_rgb(row['Image'], os.path.join(ai_dir, f"def_ai_{c_ai:05d}.jpg"))
                    c_ai += 1
        except Exception as e:
            print(f"[!] Defactify streaming terhenti: {e}")

    real_files = sorted(glob.glob(f"{real_dir}/*"))
    ai_files = sorted(glob.glob(f"{ai_dir}/*"))
    if target_cnt != float('inf'):
        real_files = real_files[:target_cnt]
        ai_files = ai_files[:target_cnt]

    records = [{'image_path': p, 'label': 0, 'source': 'Defactify'} for p in real_files] + \
              [{'image_path': p, 'label': 1, 'source': 'Defactify'} for p in ai_files]
    print(f"[+] Defactify: {len(real_files):,} Real | {len(ai_files):,} AI-Generated.")
    return records


def load_thunderbeast_dataset(cache_dir: str = "data/thunderbeast_cache", n_samples_per_class: int = 4000) -> List[Dict]:
    """Pemuat Thunderbeast (label 0 = AiArtData -> Fake, label 1 = RealArt -> Real)."""
    real_dir = os.path.join(cache_dir, "real")
    ai_dir = os.path.join(cache_dir, "ai")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    c_real = count_images_in_dir(real_dir)
    c_ai = count_images_in_dir(ai_dir)
    target_cnt = float('inf') if (n_samples_per_class is None or n_samples_per_class <= 0) else n_samples_per_class

    if min(c_real, c_ai) < target_cnt:
        print(f"[*] Mengambil Thunderbeast dari Hugging Face...")
        try:
            if load_dataset is None:
                raise ImportError("Library 'datasets' belum terinstal. Jalankan: pip install datasets")
            ds = load_dataset('Thunderbeast/AI-Generated-vs-Real-Images-Datasets', split='train')
            seen_r, seen_f = 0, 0
            for row in tqdm(ds, desc="Thunderbeast"):
                if c_real >= target_cnt and c_ai >= target_cnt:
                    break
                raw_lbl = int(row['label'])
                if raw_lbl == 1:
                    seen_r += 1
                    if seen_r <= c_real or c_real >= target_cnt:
                        continue
                    safe_save_rgb(row['image'], os.path.join(real_dir, f"tb_real_{c_real:05d}.jpg"))
                    c_real += 1
                elif raw_lbl == 0:
                    seen_f += 1
                    if seen_f <= c_ai or c_ai >= target_cnt:
                        continue
                    safe_save_rgb(row['image'], os.path.join(ai_dir, f"tb_ai_{c_ai:05d}.jpg"))
                    c_ai += 1
        except Exception as e:
            print(f"[!] Thunderbeast streaming terhenti: {e}")

    real_files = sorted(glob.glob(f"{real_dir}/*"))
    ai_files = sorted(glob.glob(f"{ai_dir}/*"))
    if target_cnt != float('inf'):
        real_files = real_files[:target_cnt]
        ai_files = ai_files[:target_cnt]

    records = [{'image_path': p, 'label': 0, 'source': 'Thunderbeast'} for p in real_files] + \
              [{'image_path': p, 'label': 1, 'source': 'Thunderbeast'} for p in ai_files]
    print(f"[+] Thunderbeast: {len(real_files):,} Real | {len(ai_files):,} AI-Art.")
    return records


def load_imd2020_dataset(cache_dir: str = "data/imd2020_cache", n_samples_per_class: int = 2000) -> List[Dict]:
    """Pemuat IMD2020 in-the-wild realistic tampering."""
    os.makedirs(cache_dir, exist_ok=True)
    existing_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.lower().endswith(VALID_EXTS)]
    target_cnt = float('inf') if (n_samples_per_class is None or n_samples_per_class <= 0) else n_samples_per_class

    if len(existing_files) < (target_cnt * 2 if target_cnt != float('inf') else 2000):
        print("[*] Mengunduh IMD2020 dari Hugging Face ('ThaneJoss/IMD2020')...")
        try:
            if load_dataset is None:
                raise ImportError("Library 'datasets' belum terinstal. Jalankan: pip install datasets")
            ds = load_dataset('ThaneJoss/IMD2020', split='full')
            count = len(existing_files)
            for i, row in enumerate(tqdm(ds, desc="IMD2020")):
                if target_cnt != float('inf') and count >= target_cnt * 2:
                    break
                lbl = int(row['label'])
                save_p = os.path.join(cache_dir, f"imd_{i:05d}_{lbl}.jpg")
                if not os.path.exists(save_p):
                    safe_save_rgb(row['image'], save_p)
                count += 1
            existing_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.lower().endswith(VALID_EXTS)]
        except Exception as e:
            print(f"[!] Warning IMD2020 download: {e}")

    real_f = [p for p in existing_files if p.endswith('_0.jpg')]
    fake_f = [p for p in existing_files if p.endswith('_1.jpg')]
    if target_cnt != float('inf'):
        real_f = real_f[:target_cnt]
        fake_f = fake_f[:target_cnt]

    records = [{'image_path': p, 'label': 0, 'source': 'IMD2020'} for p in real_f] + \
              [{'image_path': p, 'label': 1, 'source': 'IMD2020'} for p in fake_f]
    print(f"[+] IMD2020: {len(real_f):,} Real | {len(fake_f):,} Tampered.")
    return records


# ==============================================================================
# 4. PER-SOURCE BALANCING & HOLDOUT
# ==============================================================================

def balance_two_classes(df_subset: pd.DataFrame, seed: int = 42) -> Optional[pd.DataFrame]:
    r = df_subset[df_subset['label'] == 0]
    f = df_subset[df_subset['label'] == 1]
    n = min(len(r), len(f))
    if n == 0:
        return None
    return pd.concat([r.sample(n=n, random_state=seed), f.sample(n=n, random_state=seed)])


def prepare_balanced_data(all_records: List[Dict], max_per_class: int = -1, holdout_source: Optional[str] = None, seed: int = 42):
    df = pd.DataFrame(all_records).drop_duplicates(subset=['image_path'])

    holdout_df = None
    if holdout_source and holdout_source in df['source'].values:
        holdout_raw = df[df['source'] == holdout_source]
        holdout_df = balance_two_classes(holdout_raw, seed=seed)
        df = df[df['source'] != holdout_source]
        print(f"\n[🌍] Holdout Lintas-Dataset: {holdout_source} ({len(holdout_df):,} citra) DIPISAHKAN untuk uji generalisasi.")

    balanced_parts = []
    for src, group in df.groupby('source'):
        b = balance_two_classes(group, seed=seed)
        if b is None:
            print(f"[!] Sumber {src} dilewati: tidak memiliki kedua kelas.")
            continue
        balanced_parts.append(b)

    assert len(balanced_parts) > 0, "ERROR: Tidak ada data yang memenuhi syarat seimbang 50:50!"
    df_balanced = pd.concat(balanced_parts)

    if max_per_class is not None and max_per_class > 0 and (len(df_balanced) // 2) > max_per_class:
        frac = max_per_class / (len(df_balanced) / 2)
        df_balanced = df_balanced.groupby(['source', 'label']).sample(frac=frac, random_state=seed)

    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Stratified per (source, label)
    strata = df_balanced['source'] + '_' + df_balanced['label'].astype(str)
    train_df, temp_df = train_test_split(df_balanced, test_size=0.20, stratify=strata, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=strata.loc[temp_df.index], random_state=seed)

    print("\n" + "=" * 65)
    print(f"📊 DATASET FINAL SEIMBANG ({len(df_balanced):,} Total Citra):")
    print(f"   • Real (0) : {(df_balanced['label'] == 0).sum():,} citra")
    print(f"   • Fake (1) : {(df_balanced['label'] == 1).sum():,} citra")
    print(f"   - Split: Train = {len(train_df):,} | Val = {len(val_df):,} | Test = {len(test_df):,}")
    print("\nRincian Citra per Sumber & Label:")
    print(df_balanced.groupby(['source', 'label']).size().unstack(fill_value=0))
    print("=" * 65)

    return train_df, val_df, test_df, holdout_df


# ==============================================================================
# 5. TRAINING & CALIBRATION PIPELINE
# ==============================================================================

class ModelEMA:
    """Exponential Moving Average (EMA) untuk menstabilkan bobot model dan mencegah loss bouncing."""
    def __init__(self, model, decay=0.995):
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def state_dict(self):
        return self.ema_model.state_dict()


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, ema=None, aux_weight=0.3, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = (scaler is not None and device.type == 'cuda')

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels, _ in pbar:
        if isinstance(images, (tuple, list)):
            imgs_g = images[0].to(device, non_blocking=True)
            imgs_p = images[1].to(device, non_blocking=True)
            batch_sz = imgs_g.size(0)
        else:
            imgs_g = images.to(device, non_blocking=True)
            imgs_p = None
            batch_sz = imgs_g.size(0)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            main_out, aux_rgb, aux_for, _ = model(imgs_g, imgs_p, return_all=True)
            loss = criterion(main_out, labels) + aux_weight * (criterion(aux_rgb, labels) + criterion(aux_for, labels))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_sz
        preds = torch.argmax(main_out, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total*100:.2f}%"})

    return total_loss / total, (correct / total) * 100.0


def collect_logits(model, loader, device):
    model.eval()
    logits_list, labels_list, sources_list, gates_list = [], [], [], []
    with torch.no_grad():
        for images, labels, sources in tqdm(loader, desc="Inferensi", leave=False):
            if isinstance(images, (tuple, list)):
                imgs_g = images[0].to(device, non_blocking=True)
                imgs_p = images[1].to(device, non_blocking=True)
            else:
                imgs_g = images.to(device, non_blocking=True)
                imgs_p = None
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out, gate = model(imgs_g, imgs_p, return_gate=True)
            logits_list.append(out.float().cpu())
            labels_list.append(labels)
            sources_list.extend(sources)
            gates_list.extend(gate.float().cpu().tolist())

    return torch.cat(logits_list), torch.cat(labels_list), np.array(sources_list), np.array(gates_list)


def calibrate_temperature_and_threshold(val_logits: torch.Tensor, val_labels: torch.Tensor, target_fpr: float = 0.05):
    """Temperature Scaling & Balanced Macro-F1 Thresholding."""
    log_T = torch.zeros(1, requires_grad=True)
    opt_T = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200)

    def closure():
        opt_T.zero_grad()
        loss = F.cross_entropy(val_logits / log_T.exp(), val_labels)
        loss.backward()
        return loss

    opt_T.step(closure)
    temperature = float(torch.clamp(log_T.exp(), 0.1, 5.0).item())

    val_y = val_labels.numpy()
    val_probs = torch.softmax(val_logits / temperature, dim=1)[:, 1].numpy()

    # 1. Ambang Optimal Seimbang (Maximize Macro-F1)
    grid = np.linspace(0.35, 0.75, 81)
    f1_grid = [f1_score(val_y, (val_probs >= t).astype(int), average='macro', zero_division=0) for t in grid]
    best_idx = int(np.argmax(f1_grid))
    threshold_balanced = float(grid[best_idx])

    # 2. Ambang Konservatif (Target FPR <= target_fpr)
    real_probs = val_probs[val_y == 0]
    thr_fpr = float(np.quantile(real_probs, 1.0 - target_fpr)) if len(real_probs) > 0 else 0.75
    threshold_conservative = float(np.clip(thr_fpr, 0.50, 0.95))

    return temperature, threshold_balanced, threshold_conservative


def evaluate_report(logits: torch.Tensor, labels: torch.Tensor, sources: np.ndarray,
                    temperature: float = 1.0, threshold: float = 0.5, thr_conservative: float = None,
                    title: str = "HASIL EVALUASI"):
    y = labels.numpy()
    probs = torch.softmax(logits / temperature, dim=1)[:, 1].numpy()

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    comp_list = [("Ambang Default 0.500", (probs >= 0.5).astype(int)),
                 (f"Ambang Seimbang {threshold:.3f}", (probs >= threshold).astype(int))]
    if thr_conservative is not None:
        comp_list.append((f"Ambang Konservatif {thr_conservative:.3f}", (probs >= thr_conservative).astype(int)))

    for name, pred in comp_list:
        fpr = ((y == 0) & (pred == 1)).sum() / max(1, (y == 0).sum()) * 100.0
        tpr = ((y == 1) & (pred == 1)).sum() / max(1, (y == 1).sum()) * 100.0
        acc = np.mean(pred == y) * 100.0
        f1 = f1_score(y, pred, average='macro', zero_division=0) * 100.0
        print(f"{name:32s} | Acc={acc:6.2f}% | Macro-F1={f1:6.2f}% | FPR(Real->Fake)={fpr:5.2f}% | TPR(Fake Detected)={tpr:5.2f}%")

    auc_score = roc_auc_score(y, probs) * 100.0 if len(np.unique(y)) == 2 else float('nan')
    print(f"ROC-AUC (Independen Ambang)   : {auc_score:.2f}%")

    pred_final = (probs > threshold).astype(int)
    print("\nConfusion Matrix (@Ambang Terkalibrasi):")
    print(confusion_matrix(y, pred_final))
    print("\nLaporan Klasifikasi Detail:")
    print(classification_report(y, pred_final, target_names=CLASS_NAMES, digits=4, zero_division=0))

    rows = []
    for src in np.unique(sources):
        mask = (sources == src)
        sub_y, sub_pred, sub_prob = y[mask], pred_final[mask], probs[mask]
        fpr = ((sub_y == 0) & (sub_pred == 1)).sum() / max(1, (sub_y == 0).sum()) * 100.0
        tpr = ((sub_y == 1) & (sub_pred == 1)).sum() / max(1, (sub_y == 1).sum()) * 100.0
        auc = roc_auc_score(sub_y, sub_prob) * 100.0 if len(np.unique(sub_y)) == 2 else float('nan')
        rows.append({
            'Sumber': src, 'N': int(mask.sum()), 'Acc (%)': np.mean(sub_pred == sub_y) * 100.0,
            'FPR (%)': fpr, 'TPR (%)': tpr, 'AUC (%)': auc
        })
    print("Breakdown Performa per Sumber Dataset:")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("=" * 75)


# ==============================================================================
# 6. MAIN RUNNER
# ==============================================================================

def run_training(args):
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"[*] Perangkat Pelatihan: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if args.all_data:
        args.max_samples_per_class = -1
        args.casia_samples = -1
        args.defactify_samples = -1
        args.thunderbeast_samples = -1
        args.imd2020_samples = -1
        print("[🔥] Mode SEMUA DATA: Menggunakan 100% seluruh citra yang tersedia tanpa batas kuota!")

    # 1. Kumpulkan Data dari Semua Sumber
    all_records = []
    if args.use_casia:
        all_records.extend(load_casia2_dataset(args.casia_dir, n_samples_per_class=args.casia_samples, seed=args.seed))
    if args.use_defactify:
        all_records.extend(load_defactify_dataset(n_samples_per_class=args.defactify_samples))
    if args.use_thunderbeast:
        all_records.extend(load_thunderbeast_dataset(n_samples_per_class=args.thunderbeast_samples))
    if args.use_imd2020:
        all_records.extend(load_imd2020_dataset(n_samples_per_class=args.imd2020_samples))

    if len(all_records) == 0:
        print("[!] Gagal memuat data citra dari dataset manapun. Periksa direktori atau koneksi.")
        return

    # 2. Balancing & Split
    holdout_src = None if args.holdout == "None" else args.holdout
    train_df, val_df, test_df, holdout_df = prepare_balanced_data(
        all_records, max_per_class=args.max_samples_per_class, holdout_source=holdout_src, seed=args.seed
    )

    if args.crop_mode == 'dual_scale':
        train_g_tf, train_p_tf = get_dual_scale_transforms(img_size=args.img_size, is_train=True, jpeg_p=args.jpeg_aug_p)
        eval_g_tf, eval_p_tf = get_dual_scale_transforms(img_size=args.img_size, is_train=False)

        train_ds = DualScaleForensicDataset(train_df, global_tf=train_g_tf, patch_tf=train_p_tf, is_eval=False)
        val_ds = DualScaleForensicDataset(val_df, global_tf=eval_g_tf, patch_tf=eval_p_tf, is_eval=True)
        test_ds = DualScaleForensicDataset(test_df, global_tf=eval_g_tf, patch_tf=eval_p_tf, is_eval=True)
        print(f"[+] Mode Preprocessing: DUAL-SCALE (Global Resized 224x224 + Salient Micro Patch 1:1 Resolusi Murni)")
    else:
        use_crop = (args.crop_mode == 'crop')
        train_tf = get_v5_transforms(img_size=args.img_size, is_train=True, use_crop=use_crop, jpeg_p=args.jpeg_aug_p)
        eval_tf = get_v5_transforms(img_size=args.img_size, is_train=False, use_crop=use_crop)

        train_ds = UniversalForensicDataset(train_df, transform=train_tf)
        val_ds = UniversalForensicDataset(val_df, transform=eval_tf)
        test_ds = UniversalForensicDataset(test_df, transform=eval_tf)
        print(f"[+] Mode Preprocessing: SINGLE-SCALE ({args.crop_mode})")

    num_workers = args.num_workers if os.name != 'nt' else 0
    loader_kw = dict(batch_size=args.batch_size, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    # 3. Model
    print(f"\n[*] Menginisialisasi Model AIFusionNet-DualScale V5 ({args.crop_mode} {args.img_size}px)...")
    model = build_v5_model(num_classes=2, dropout_p=args.dropout).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Total Parameter: {total_params:,} (~{total_params*4 / (1024**2):.2f} MB float32)")

    # 4. Optimizer, Scheduler, Loss, Scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    min_lr_ratio = 1e-6 / args.lr

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if (args.amp and device.type == 'cuda') else None

    # 5. Training Loop with Model EMA and Early Stopping
    best_val_f1 = 0.0
    epochs_no_improve = 0
    print(f"\n🚀 Memulai Pelatihan ({args.epochs} Epochs, EMA={args.ema_decay}, Early Stopping Patience={args.patience})...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device,
                                            ema=ema, aux_weight=0.3, grad_clip=args.grad_clip)

        # Evaluasi Validasi menggunakan Bobot Model EMA yang Dihaluskan
        val_logits, val_labels, _, val_gates = collect_logits(ema.ema_model, val_loader, device)
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].numpy()
        val_targets = val_labels.numpy()
        val_preds = (val_probs >= 0.5).astype(int)

        val_loss = F.cross_entropy(val_logits, val_labels).item()
        val_acc = np.mean(val_preds == val_targets) * 100.0
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0) * 100.0
        val_auc = roc_auc_score(val_targets, val_probs) * 100.0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Val(EMA): Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.2f}%, AUC={val_auc:.2f}%, Avg Gate={np.mean(val_gates):.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'arch': 'aifusionnet_v5_universal',
                'config': {
                    'arch_variant': 'dual_scale' if args.crop_mode == 'dual_scale' else args.crop_mode,
                    'img_size': args.img_size,
                    'crop_mode': 'salient_patch' if args.crop_mode == 'dual_scale' else args.crop_mode,
                    'mean': IMAGENET_MEAN,
                    'std': IMAGENET_STD,
                    'class_names': CLASS_NAMES,
                    'dropout': args.dropout,
                    'holdout': holdout_src,
                    'ema_decay': args.ema_decay,
                }
            }, args.output_path)
            print(f"  ⭐ Checkpoint Disimpan (EMA Weights): {args.output_path} (Val F1: {val_f1:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  ⏹️ Early stopping aktif: Val F1 tidak meningkat selama {args.patience} epoch berturut-turut.")
                break

    # 6. Calibration & Final Testing
    print("\n" + "=" * 65)
    print("🎯 KALIBRASI POST-TRAINING (Temperature Scaling & Balanced Macro-F1 Threshold)")
    print("=" * 65)
    try:
        checkpoint = torch.load(args.output_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.output_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_logits, val_labels, _, _ = collect_logits(model, val_loader, device)
    temperature, threshold_bal, threshold_cons = calibrate_temperature_and_threshold(val_logits, val_labels, target_fpr=args.target_fpr)
    print(f"[+] Calibrated Temperature T       = {temperature:.3f}")
    print(f"[★] Ambang Seimbang Optimal (F1)   = {threshold_bal:.3f} -> DIPILIH UTAMA")
    print(f"[i] Ambang Konservatif (FPR<={args.target_fpr:.0%}) = {threshold_cons:.3f}")

    # Update checkpoint with calibration
    checkpoint.update({
        'temperature': temperature,
        'threshold': threshold_bal,
        'threshold_balanced': threshold_bal,
        'threshold_conservative': threshold_cons,
        'target_fpr': args.target_fpr
    })
    torch.save(checkpoint, args.output_path)

    # In-Distribution Test Evaluation
    test_logits, test_labels, test_sources, _ = collect_logits(model, test_loader, device)
    evaluate_report(test_logits, test_labels, test_sources, temperature=temperature, threshold=threshold_bal,
                    thr_conservative=threshold_cons, title="EVALUASI TEST SET INDEPENDEN (In-Distribution)")

    # Holdout Cross-Dataset Evaluation
    if holdout_df is not None:
        if args.crop_mode == 'dual_scale':
            holdout_ds = DualScaleForensicDataset(holdout_df, global_tf=eval_g_tf, patch_tf=eval_p_tf, is_eval=True)
        else:
            holdout_ds = UniversalForensicDataset(holdout_df, transform=eval_tf)
        holdout_loader = DataLoader(holdout_ds, shuffle=False, **loader_kw)
        h_logits, h_labels, h_sources, _ = collect_logits(model, holdout_loader, device)
        evaluate_report(h_logits, h_labels, h_sources, temperature=temperature, threshold=threshold_bal,
                        thr_conservative=threshold_cons, title=f"EVALUASI HOLDOUT LINTAS-DATASET: {holdout_src} (Zero-Shot Generalization)")

    # JPEG Robustness Stress Test
    if args.test_jpeg:
        if args.crop_mode == 'dual_scale':
            jpeg_patch_tf = transforms.Compose([FixedJPEG(75), transforms.ToTensor(), norm])
            jpeg_ds = DualScaleForensicDataset(test_df, global_tf=eval_g_tf, patch_tf=jpeg_patch_tf, is_eval=True)
        else:
            jpeg_tf = transforms.Compose([FixedJPEG(75)] + list(eval_tf.transforms))
            jpeg_ds = UniversalForensicDataset(test_df, transform=jpeg_tf)
        jpeg_loader = DataLoader(jpeg_ds, shuffle=False, **loader_kw)
        j_logits, j_labels, j_sources, _ = collect_logits(model, jpeg_loader, device)
        evaluate_report(j_logits, j_labels, j_sources, temperature=temperature, threshold=threshold_bal,
                        thr_conservative=threshold_cons, title="UJI KETAHANAN: Test Set Dikompresi Ulang JPEG Q=75")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pelatihan AIFusionNet-DualScale V5 (V5.3)")
    parser.add_argument('--epochs', type=int, default=20, help="Jumlah epoch (default: 20)")
    parser.add_argument('--batch-size', type=int, default=32, help="Ukuran batch (default: 32)")
    parser.add_argument('--lr', type=float, default=1.5e-4, help="Learning rate (default: 1.5e-4)")
    parser.add_argument('--weight-decay', type=float, default=1e-3, help="Weight decay L2 (default: 1e-3)")
    parser.add_argument('--dropout', type=float, default=0.40, help="Dropout rate (default: 0.40)")
    parser.add_argument('--warmup-epochs', type=int, default=3, help="Warmup epochs (default: 3)")
    parser.add_argument('--patience', type=int, default=6, help="Early stopping patience (default: 6)")
    parser.add_argument('--ema-decay', type=float, default=0.995, help="Decay untuk Model EMA (default: 0.995)")
    parser.add_argument('--grad-clip', type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument('--label-smoothing', type=float, default=0.10, help="Label smoothing (default: 0.10)")
    parser.add_argument('--crop-mode', type=str, default='dual_scale', choices=['dual_scale', 'crop', 'resize'], help="Mode preprocessing (dual_scale / crop / resize)")
    parser.add_argument('--jpeg-aug-p', type=float, default=0.8, help="Peluang augmentasi JPEG kedua kelas")
    parser.add_argument('--target-fpr', type=float, default=0.05, help="Target false positive rate untuk kalibrasi ambang")
    parser.add_argument('--holdout', type=str, default='None', choices=['None', 'CASIA2', 'Defactify', 'Thunderbeast', 'IMD2020'], help="Holdout source untuk zero-shot test")
    parser.add_argument('--test-jpeg', action='store_true', default=True, help="Jalankan uji ketahanan kompresi JPEG Q=75")
    parser.add_argument('--use-casia', action=argparse.BooleanOptionalAction, default=True, help="Aktifkan/nonaktifkan CASIA v2.0")
    parser.add_argument('--use-defactify', action=argparse.BooleanOptionalAction, default=True, help="Aktifkan/nonaktifkan Defactify")
    parser.add_argument('--use-thunderbeast', action=argparse.BooleanOptionalAction, default=True, help="Aktifkan/nonaktifkan Thunderbeast")
    parser.add_argument('--use-imd2020', action=argparse.BooleanOptionalAction, default=False, help="Aktifkan/nonaktifkan IMD2020")
    parser.add_argument('--all-data', action='store_true', default=False, help="Gunakan 100%% seluruh data tanpa batas kuota")
    parser.add_argument('--casia-dir', type=str, default='data/casia2', help="Direktori CASIA v2.0")
    parser.add_argument('--casia-samples', type=int, default=4000, help="Sampel CASIA v2.0 per kelas (-1 untuk semua)")
    parser.add_argument('--defactify-samples', type=int, default=4000, help="Sampel Defactify per kelas (-1 untuk semua)")
    parser.add_argument('--thunderbeast-samples', type=int, default=4000, help="Sampel Thunderbeast per kelas (-1 untuk semua)")
    parser.add_argument('--imd2020-samples', type=int, default=2000, help="Sampel IMD2020 per kelas (-1 untuk semua)")
    parser.add_argument('--max-samples-per-class', type=int, default=12000, help="Total sampel seimbang per kelas (-1 untuk semua)")
    parser.add_argument('--img-size', type=int, default=224, help="Resolusi spasial / crop")
    parser.add_argument('--num-workers', type=int, default=2, help="Jumlah worker DataLoader")
    parser.add_argument('--output-path', type=str, default='best_aifusionnet_v5_universal.pth', help="Path file checkpoint")
    parser.add_argument('--amp', action='store_true', default=True, help="Automatic Mixed Precision")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda / cpu)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_training(args)
