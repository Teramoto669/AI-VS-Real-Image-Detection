"""
AIForensicNet V4 Universal - Standalone Quad-Dataset Training Pipeline
Mendukung 4 Dataset Forensik & Generatif AI Sekaligus:
1. CASIA v2.0   : Manipulasi Spasial (Photoshop Splicing & Copy-Move)
2. Defactify    : Citra AI Modern (Midjourney v6, DALL-E 3, SD 2.1, SDXL, SD 3) vs Real (MS-COCO)
3. Thunderbeast : AI-Generated Art vs Real Art
4. IMD2020      : Manipulasi Realistis In-The-Wild (WACV 2020)

Arsitektur: AIForensicNet V4 Universal (Two-Stream Dual-Branch dengan Learnable Gated Cross-Attention)
"""

import os
import sys
import time
import io
import zipfile
import random
import argparse
import warnings
from io import BytesIO

warnings.filterwarnings('ignore', message='.*Palette images with Transparency.*')
warnings.filterwarnings('ignore', message='.*xFormers is not available.*')

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

from model_custom import AIForensicNetV4Universal


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al.) untuk menangani sampel manipulasi yang sulit (hard negatives):
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Secara otomatis meredam loss sampel mudah dan memusatkan gradien pada citra manipulasi halus.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ==============================================================================
# 1. FORENSIC AUGMENTATION MODULES
# ==============================================================================

class RandomJPEGCompression:
    """
    Simulasi kompresi lossy JPEG acak pada skala resolusi asli.
    Melatih model agar kebal terhadap kompresi media sosial (WhatsApp, WebP, IG)
    dan tidak keliru mendeteksi artefak kompresi sebagai pemalsuan AI.
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
    Mempertahankan resolusi murni citra (tanpa bilinear downsampling prematur).
    Random Crop pada training, Center Crop pada evaluasi/validasi.
    """
    def __init__(self, size=224, is_train=True):
        self.size = size
        self.is_train = is_train

    def __call__(self, img):
        w, h = img.size
        if min(w, h) < self.size:
            scale = float(self.size) / float(min(w, h))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            w, h = img.size

        if self.is_train:
            left = random.randint(0, w - self.size)
            top = random.randint(0, h - self.size)
        else:
            left = (w - self.size) // 2
            top = (h - self.size) // 2

        return img.crop((left, top, left + self.size, top + self.size))


class QuadForensicDataset(Dataset):
    """
    PyTorch Dataset untuk Quad Dataset Universal Forensics dengan Dual-Scale Asymmetric Input:
    - img_global : Resized ke 224x224 (Branch 1: Membaca semantik utuh, pencahayaan, difusi AI)
    - img_crop   : Native Crop 224x224 (Branch 2: Membaca frekuensi murni, diskontinuitas tepi splicing)
    """
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.crop_op = NativeResolutionCrop(size=224, is_train=is_train)
        self.jpeg_aug = RandomJPEGCompression(quality_range=(60, 95), p=0.45) if is_train else None
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        try:
            pil_img = Image.open(img_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
        except Exception:
            pil_img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        # Scale 1: Global Full-Scene View untuk Branch 1 (AI Generatif)
        img_global_pil = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
        flip = self.is_train and (random.random() < 0.5)
        if flip:
            img_global_pil = img_global_pil.transpose(Image.FLIP_LEFT_RIGHT)
        t_global = self.normalize(self.to_tensor(img_global_pil))

        # Scale 2: Native-Resolution Micro-Crop untuk Branch 2 (Forensik Frekuensi & Splicing)
        img_crop_pil = self.crop_op(pil_img)
        if flip:
            img_crop_pil = img_crop_pil.transpose(Image.FLIP_LEFT_RIGHT)
        if self.jpeg_aug:
            img_crop_pil = self.jpeg_aug(img_crop_pil)
        t_crop = self.normalize(self.to_tensor(img_crop_pil))

        label = int(row['label'])
        return t_global, t_crop, label, row['source'], row['manipulation']


# ==============================================================================
# 2. DATASET LOADERS (CASIA2, DEFACTIFY, THUNDERBEAST, IMD2020)
# ==============================================================================

VALID_EXTS = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp', '.webp')


def load_casia2_dataset(data_dir="data/casia2", zip_path="CASIA2.zip"):
    """
    Pemuat CASIA v2.0 (Authentic vs Splicing & Copy-Move)
    Kaggle: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
    """
    os.makedirs(data_dir, exist_ok=True)

    def locate_dirs(base_path):
        for root, dirs, _ in os.walk(base_path):
            dir_names_lower = [d.lower() for d in dirs]
            if 'au' in dir_names_lower and 'tp' in dir_names_lower:
                au = os.path.join(root, dirs[dir_names_lower.index('au')])
                tp = os.path.join(root, dirs[dir_names_lower.index('tp')])
                return au, tp
        return None, None

    au_dir, tp_dir = locate_dirs(data_dir)
    if au_dir is None or tp_dir is None:
        print("[*] Mengunduh dataset CASIA v2.0 ('divg07/casia-20-image-tampering-detection-dataset')...")
        # 1. Coba unduh via kagglehub
        try:
            import kagglehub
            k_path = kagglehub.dataset_download("divg07/casia-20-image-tampering-detection-dataset")
            au_dir, tp_dir = locate_dirs(k_path)
            if au_dir and tp_dir:
                print(f"[+] Berhasil mengunduh CASIA v2.0 via KaggleHub di: {k_path}")
        except Exception as ex:
            print(f"[!] Info KaggleHub download: {ex}")

        # 2. Coba via Google Drive mirror jika kagglehub belum terautentikasi
        if au_dir is None or tp_dir is None:
            if not os.path.exists(zip_path):
                try:
                    import gdown
                    DRIVE_FILE_ID = "1IDUgcoUeonBxx2rASX-_QwV9fhbtqdY8"
                    gdown.download(id=DRIVE_FILE_ID, output=zip_path, quiet=False)
                except Exception as e:
                    print(f"[!] Gdown download error: {e}")

            if os.path.exists(zip_path):
                print("[*] Mengekstrak arsip CASIA2.zip...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                au_dir, tp_dir = locate_dirs(data_dir)

    records = []
    if au_dir and tp_dir and os.path.exists(au_dir) and os.path.exists(tp_dir):
        real_files = [os.path.join(au_dir, f) for f in os.listdir(au_dir) if f.lower().endswith(VALID_EXTS)]
        fake_files = [os.path.join(tp_dir, f) for f in os.listdir(tp_dir) if f.lower().endswith(VALID_EXTS)]
        print(f"[+] CASIA v2.0 Siap: {len(real_files)} Real (Au) | {len(fake_files)} Fake (Tp)")

        for p in real_files:
            records.append({'image_path': p, 'label': 0, 'label_name': 'Real', 'manipulation': 'Authentic Camera Photo', 'source': 'CASIA v2.0'})
        for p in fake_files:
            fname = os.path.basename(p)
            manip = 'Splicing' if 'Tp_S' in fname else ('Copy-Move' if 'Tp_D' in fname else 'Tampered')
            records.append({'image_path': p, 'label': 1, 'label_name': 'Fake', 'manipulation': manip, 'source': 'CASIA v2.0'})
    else:
        print("[!] Folder CASIA v2.0 belum ditemukan di disk. Melewati CASIA2...")

    return records


def load_defactify_dataset(cache_dir="data/defactify_cache", n_samples_per_class=6000):
    """
    Pemuat Defactify (MS COCOAI: Midjourney v6, DALL-E 3, SDXL, SD 3 vs Real MS-COCO)
    HuggingFace: https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset
    """
    os.makedirs(f"{cache_dir}/real", exist_ok=True)
    os.makedirs(f"{cache_dir}/ai", exist_ok=True)

    existing_real = [os.path.join(f"{cache_dir}/real", f) for f in os.listdir(f"{cache_dir}/real") if f.lower().endswith(VALID_EXTS)]
    existing_ai = [os.path.join(f"{cache_dir}/ai", f) for f in os.listdir(f"{cache_dir}/ai") if f.lower().endswith(VALID_EXTS)]

    records = []
    for p in existing_real[:n_samples_per_class]:
        records.append({'image_path': p, 'label': 0, 'label_name': 'Real', 'manipulation': 'Authentic Photo (MS-COCO)', 'source': 'Defactify'})
    for p in existing_ai[:n_samples_per_class]:
        records.append({'image_path': p, 'label': 1, 'label_name': 'Fake', 'manipulation': 'AI-Generated (MJ/DALL-E/SD)', 'source': 'Defactify'})

    needed_real = max(0, n_samples_per_class - len(existing_real))
    needed_ai = max(0, n_samples_per_class - len(existing_ai))

    if needed_real == 0 and needed_ai == 0:
        print(f"[+] Memuat Defactify dari cache lokal: {len(records)} citra ({len(existing_real)} Real, {len(existing_ai)} AI).")
        return records

    print(f"[*] Mengambil streaming data dari Hugging Face ('Rajarshi-Roy-research/Defactify_Image_Dataset')...")
    model_names = {0: 'MS-COCO', 1: 'Stable Diffusion 2.1', 2: 'SDXL', 3: 'Stable Diffusion 3', 4: 'DALL-E 3', 5: 'Midjourney v6'}
    try:
        ds = load_dataset('Rajarshi-Roy-research/Defactify_Image_Dataset', split='train', streaming=True)
        count_real = len(existing_real)
        count_fake = len(existing_ai)
        pbar = tqdm(total=(n_samples_per_class * 2), desc="Streaming Defactify")
        pbar.update(count_real + count_fake)

        for row in ds:
            if count_real >= n_samples_per_class and count_fake >= n_samples_per_class:
                break

            lbl_a = int(row['Label_A'])
            lbl_b = int(row.get('Label_B', 0))
            m_name = model_names.get(lbl_b, 'AI Generator')

            if lbl_a == 0 and count_real < n_samples_per_class:
                save_path = f"{cache_dir}/real/defactify_real_{count_real:06d}.jpg"
                if not os.path.exists(save_path):
                    img = row['Image'].convert('RGB')
                    img.save(save_path, 'JPEG', quality=95)
                records.append({'image_path': save_path, 'label': 0, 'label_name': 'Real', 'manipulation': 'Authentic Photo (MS-COCO)', 'source': 'Defactify'})
                count_real += 1
                pbar.update(1)
            elif lbl_a == 1 and count_fake < n_samples_per_class:
                save_path = f"{cache_dir}/ai/defactify_ai_{count_fake:06d}.jpg"
                if not os.path.exists(save_path):
                    img = row['Image'].convert('RGB')
                    img.save(save_path, 'JPEG', quality=95)
                records.append({'image_path': save_path, 'label': 1, 'label_name': 'Fake', 'manipulation': f'AI-Generated ({m_name})', 'source': 'Defactify'})
                count_fake += 1
                pbar.update(1)

        pbar.close()
        print(f"[+] Defactify Siap: {count_real} Real | {count_fake} AI-Generated")
    except Exception as e:
        print(f"[!] Warning streaming Defactify: {e}. Menggunakan sampel lokal yang sudah ada.")

    return records


def load_thunderbeast_dataset(cache_dir="data/thunderbeast_cache", n_samples_per_class=6000):
    """
    Pemuat Thunderbeast (AI Art vs Real Art)
    HuggingFace: https://huggingface.co/datasets/Thunderbeast/AI-Generated-vs-Real-Images-Datasets
    """
    os.makedirs(f"{cache_dir}/real", exist_ok=True)
    os.makedirs(f"{cache_dir}/ai", exist_ok=True)

    existing_real = [os.path.join(f"{cache_dir}/real", f) for f in os.listdir(f"{cache_dir}/real") if f.lower().endswith(VALID_EXTS)]
    existing_ai = [os.path.join(f"{cache_dir}/ai", f) for f in os.listdir(f"{cache_dir}/ai") if f.lower().endswith(VALID_EXTS)]

    if len(existing_real) >= n_samples_per_class and len(existing_ai) >= n_samples_per_class:
        print(f"[+] Memuat Thunderbeast dari cache lokal: {n_samples_per_class} Real Art, {n_samples_per_class} AI Art.")
        records = []
        for p in existing_real[:n_samples_per_class]:
            records.append({'image_path': p, 'label': 0, 'label_name': 'Real', 'manipulation': 'Authentic Art', 'source': 'Thunderbeast'})
        for p in existing_ai[:n_samples_per_class]:
            records.append({'image_path': p, 'label': 1, 'label_name': 'Fake', 'manipulation': 'AI-Generated Art', 'source': 'Thunderbeast'})
        return records

    print("[*] Mengunduh/mengekstrak Thunderbeast ('Thunderbeast/AI-Generated-vs-Real-Images-Datasets')...")
    try:
        # Prioritaskan cache_dir='data' jika arrow file sudah ada
        ds = load_dataset('Thunderbeast/AI-Generated-vs-Real-Images-Datasets', cache_dir='data', split='train')
        records = []
        count_real, count_fake = len(existing_real), len(existing_ai)

        for row in tqdm(ds, desc="Menyiapkan Thunderbeast"):
            if count_real >= n_samples_per_class and count_fake >= n_samples_per_class:
                break
            raw_lbl = int(row['label'])
            # Label asli: 1 = Real, 0 = AI-Generated. Diselaraskan: 0 = Real, 1 = Fake.
            unified_lbl = 0 if raw_lbl == 1 else 1

            if unified_lbl == 0 and count_real < n_samples_per_class:
                save_path = f"{cache_dir}/real/tb_real_{count_real:06d}.jpg"
                if not os.path.exists(save_path):
                    row['image'].convert('RGB').save(save_path, 'JPEG', quality=95)
                records.append({'image_path': save_path, 'label': 0, 'label_name': 'Real', 'manipulation': 'Authentic Art', 'source': 'Thunderbeast'})
                count_real += 1
            elif unified_lbl == 1 and count_fake < n_samples_per_class:
                save_path = f"{cache_dir}/ai/tb_ai_{count_fake:06d}.jpg"
                if not os.path.exists(save_path):
                    row['image'].convert('RGB').save(save_path, 'JPEG', quality=95)
                records.append({'image_path': save_path, 'label': 1, 'label_name': 'Fake', 'manipulation': 'AI-Generated Art', 'source': 'Thunderbeast'})
                count_fake += 1

        print(f"[+] Thunderbeast Siap: {count_real} Real Art | {count_fake} AI Art")
        return records
    except Exception as e:
        print(f"[!] Warning Thunderbeast loader: {e}. Menggunakan file cache lokal...")
        records = []
        for p in existing_real:
            records.append({'image_path': p, 'label': 0, 'label_name': 'Real', 'manipulation': 'Authentic Art', 'source': 'Thunderbeast'})
        for p in existing_ai:
            records.append({'image_path': p, 'label': 1, 'label_name': 'Fake', 'manipulation': 'AI-Generated Art', 'source': 'Thunderbeast'})
        return records


def load_imd2020_dataset(cache_dir="data/imd2020_cache"):
    """
    Pemuat IMD2020 (WACV 2020: In-the-wild realistic image tampering)
    Kaggle: https://www.kaggle.com/datasets/saikotahmed/imd2020
    HuggingFace Mirror: https://huggingface.co/datasets/ThaneJoss/IMD2020
    """
    os.makedirs(cache_dir, exist_ok=True)
    existing_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.lower().endswith(VALID_EXTS)]

    if len(existing_files) >= 1500:
        print(f"[+] Memuat IMD2020 dari cache disk: {len(existing_files)} citra.")
        records = []
        for p in existing_files:
            fname = os.path.basename(p)
            try:
                lbl = int(fname.split('_')[-1].split('.')[0])
            except Exception:
                lbl = 1 if ('fake' in fname.lower() or 'tampered' in fname.lower() or 'tp' in fname.lower()) else 0
            records.append({
                'image_path': p,
                'label': lbl,
                'label_name': 'Real' if lbl == 0 else 'Fake',
                'manipulation': 'Tampered (In-the-wild)' if lbl == 1 else 'Authentic Web Photo',
                'source': 'IMD2020'
            })
        return records

    print("[*] Menyiapkan IMD2020 (Mencoba Kaggle 'saikotahmed/imd2020' & HuggingFace 'ThaneJoss/IMD2020')...")
    # 1. Coba via kagglehub ('saikotahmed/imd2020')
    try:
        import kagglehub
        k_path = kagglehub.dataset_download("saikotahmed/imd2020")
        print(f"[+] Terunduh via KaggleHub di: {k_path}")
        records = []
        for root, _, files in os.walk(k_path):
            for f in files:
                if f.lower().endswith(VALID_EXTS) and 'mask' not in f.lower():
                    full_p = os.path.join(root, f)
                    lbl = 0 if ('orig' in f.lower() or 'real' in f.lower() or 'au' in f.lower()) else 1
                    records.append({
                        'image_path': full_p,
                        'label': lbl,
                        'label_name': 'Real' if lbl == 0 else 'Fake',
                        'manipulation': 'Tampered (In-the-wild)' if lbl == 1 else 'Authentic Photo',
                        'source': 'IMD2020'
                    })
        if len(records) > 0:
            print(f"[+] IMD2020 Siap via KaggleHub: {len(records)} citra.")
            return records
    except Exception as e:
        print(f"[!] Info KaggleHub IMD2020: {e}. Menggunakan HuggingFace mirror...")

    # 2. Coba via Hugging Face ('ThaneJoss/IMD2020')
    try:
        ds = load_dataset('ThaneJoss/IMD2020', split='full')
        records = []
        for i, row in enumerate(tqdm(ds, desc="Membaca IMD2020")):
            img = row['image']
            lbl = int(row['label'])
            save_path = f"{cache_dir}/imd_{i:05d}_{lbl}.jpg"
            if not os.path.exists(save_path):
                img.convert('RGB').save(save_path, 'JPEG', quality=95)
            records.append({
                'image_path': save_path,
                'label': lbl,
                'label_name': 'Real' if lbl == 0 else 'Fake',
                'manipulation': row.get('manipulation', 'Tampered' if lbl == 1 else 'Authentic'),
                'source': 'IMD2020'
            })
        print(f"[+] IMD2020 Siap via HuggingFace: {len(records)} citra.")
        return records
    except Exception as e:
        print(f"[!] Warning IMD2020 loader: {e}. Menggunakan cache yang ada ({len(existing_files)} file).")
        return [{'image_path': p, 'label': 1, 'label_name': 'Fake', 'manipulation': 'Tampered', 'source': 'IMD2020'} for p in existing_files]


# ==============================================================================
# 3. STRATIFIED MULTI-DOMAIN DATASET BALANCER
# ==============================================================================

def build_balanced_quad_dataset(records, max_per_class=15000, seed=42):
    """Menyeimbangkan data 50% Real dan 50% Fake secara merata antar dataset."""
    df = pd.DataFrame(records)
    print("\n" + "=" * 65)
    print("[*] Ringkasan Komposisi Citra Mentah per Sumber:")
    print(df.groupby(['source', 'label_name']).size())
    print("=" * 65)

    df_real = df[df['label'] == 0]
    df_fake = df[df['label'] == 1]

    n_target = min(len(df_real), len(df_fake))
    if max_per_class is not None and max_per_class > 0:
        n_target = min(n_target, max_per_class)

    df_real_sampled = df_real.sample(n=n_target, random_state=seed)
    df_fake_sampled = df_fake.sample(n=n_target, random_state=seed)
    df_final = pd.concat([df_real_sampled, df_fake_sampled]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"\n[+] Dataset Final Seimbang ({len(df_final):,} total citra):")
    print(f"    - Real (0) : {(df_final['label'] == 0).sum():,} citra")
    print(f"    - Fake (1) : {(df_final['label'] == 1).sum():,} citra")
    print("\n[+] Rincian Sebaran Sub-Manipulasi:")
    print(df_final['manipulation'].value_counts())
    print("=" * 65)
    return df_final


# ==============================================================================
# 4. TRAINING & EVALUATION PIPELINE
# ==============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_quad_model(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    print("=" * 65)
    print("🚀 PELATIHAN AIFORENSICNET V4 UNIVERSAL - QUAD DATASET SOTA")
    print(f"Device Utama     : {device}")
    if device.type == 'cuda':
        print(f"GPU Name         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Tersedia    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 65)

    # 1. Pemuatan 4 Dataset
    all_records = []
    print("\n--- 1. Memuat CASIA v2.0 ---")
    all_records.extend(load_casia2_dataset(data_dir=args.casia_dir))

    print("\n--- 2. Memuat Defactify (MS COCOAI) ---")
    all_records.extend(load_defactify_dataset(n_samples_per_class=args.defactify_samples))

    print("\n--- 3. Memuat Thunderbeast ---")
    all_records.extend(load_thunderbeast_dataset(n_samples_per_class=args.thunderbeast_samples))

    print("\n--- 4. Memuat IMD2020 ---")
    all_records.extend(load_imd2020_dataset())

    if len(all_records) == 0:
        print("[!] Gagal memuat citra dari dataset manapun. Periksa koneksi internet atau path lokal.")
        return

    # 2. Balancing & Split 80:10:10
    df_balanced = build_balanced_quad_dataset(all_records, max_per_class=args.max_samples_per_class, seed=args.seed)

    train_df, temp_df = train_test_split(df_balanced, test_size=0.20, stratify=df_balanced['label'], random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=args.seed)

    print(f"\n[+] Distribusi Split Data:")
    print(f"    - Training   : {len(train_df):,} citra (80%)")
    print(f"    - Validasi   : {len(val_df):,} citra (10%)")
    print(f"    - Pengujian  : {len(test_df):,} citra (10%)")

    # 3. Transformasi Forensik Dual-Scale Asymmetric
    train_ds = QuadForensicDataset(train_df, is_train=True)
    val_ds = QuadForensicDataset(val_df, is_train=False)
    test_ds = QuadForensicDataset(test_df, is_train=False)

    num_workers = 4 if os.name != 'nt' else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    # 4. Inisialisasi Model AIForensicNet V4 Universal
    print("\n[*] Menginisialisasi Arsitektur AIForensicNet V4 Universal (Two-Stream Dual-Branch Gated)...")
    model = AIForensicNetV4Universal(num_classes=2, dropout_p=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Total Parameter Model: {total_params:,} parameter (~{total_params*4 / (1024**2):.2f} MB float32)")

    # 5. Konfigurasi Loss, Optimizer, dan Scaler
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if getattr(args, 'loss', 'focal') == 'focal':
        criterion = FocalLoss(alpha=weights_tensor, gamma=2.0, label_smoothing=0.05)
        print("[+] Loss Function: Focal Loss (gamma=2.0) dengan Label Smoothing & Class Balancing")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
        print("[+] Loss Function: Cross-Entropy Loss Standar")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_amp = (device.type == 'cuda' and args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 6. Training Loop
    print(f"\n[*] Memulai Pelatihan ({args.epochs} Epoch, Batch Size: {args.batch_size})...\n")
    best_val_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False)
        for img_global, img_crop, labels, _, _ in train_pbar:
            img_global = img_global.to(device, non_blocking=True)
            img_crop = img_crop.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(img_global, x_crop=img_crop)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * img_global.size(0)
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_df)
        epoch_train_acc = (np.sum(np.array(train_preds) == np.array(train_targets)) / len(train_df)) * 100

        # --- VALIDASI ---
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for img_global, img_crop, labels, _, _ in val_loader:
                img_global = img_global.to(device, non_blocking=True)
                img_crop = img_crop.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(img_global, x_crop=img_crop)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * img_global.size(0)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_df)
        epoch_val_acc = (np.sum(np.array(val_preds) == np.array(val_targets)) / len(val_df)) * 100
        epoch_val_f1 = f1_score(val_targets, val_preds, average='macro') * 100

        scheduler.step()

        # Simpan Checkpoint Terbaik
        saved_marker = ""
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            checkpoint = {
                'epoch': int(epoch),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': float(epoch_val_f1),
                'val_acc': float(epoch_val_acc),
                'dataset': 'CASIA2_DEFACTIFY_THUNDERBEAST_IMD2020',
                'arch': 'aiforensicnet_v4_universal'
            }
            torch.save(checkpoint, args.output_path)
            if args.universal_backup:
                torch.save(checkpoint, 'best_aiforensicnet_v4_universal.pth')
            saved_marker = "==> [CHECKPOINT TERSIMPAN]"

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | "
              f"Val F1: {epoch_val_f1:.2f}% {saved_marker}")

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"Pelatihan Selesai dalam {total_time/60:.2f} menit! Best Val F1: {best_val_f1:.2f}%")
    print(f"File Bobot Tersimpan: {args.output_path}")
    print("=" * 65)

    # 7. Evaluasi Komprehensif pada Test Set Independen
    print("\n📊 EVALUASI TEST SET INDEPENDEN...")
    if os.path.exists(args.output_path):
        ckpt = torch.load(args.output_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    test_preds, test_targets, test_sources, test_alphas = [], [], [], []

    with torch.no_grad():
        for img_global, img_crop, labels, sources, _ in tqdm(test_loader, desc="Testing"):
            img_global = img_global.to(device)
            img_crop = img_crop.to(device)
            outputs, alpha = model(img_global, x_crop=img_crop, return_gate=True)
            preds = outputs.argmax(dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())
            test_sources.extend(sources)
            test_alphas.extend(alpha.cpu().numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_sources = np.array(test_sources)
    test_alphas = np.array(test_alphas)

    overall_acc = (test_preds == test_targets).mean() * 100
    overall_f1 = f1_score(test_targets, test_preds, average='macro') * 100
    cm = confusion_matrix(test_targets, test_preds)

    print("\n" + "=" * 65)
    print(f"HASIL AKHIR TEST SET ({len(test_targets):,} Citra):")
    print(f"- Akurasi Keseluruhan : {overall_acc:.2f}%")
    print(f"- Macro F1-Score      : {overall_f1:.2f}%")
    print(f"- Confusion Matrix:\n{cm}")
    print("\nLaporan Klasifikasi per Kelas:")
    print(classification_report(test_targets, test_preds, target_names=['Real (0)', 'Fake/Tampered (1)'], digits=4))

    print("\n" + "-" * 65)
    print("Rincian Akurasi per Sumber Dataset:")
    for src in np.unique(test_sources):
        mask = (test_sources == src)
        src_acc = (test_preds[mask] == test_targets[mask]).mean() * 100
        src_f1 = f1_score(test_targets[mask], test_preds[mask], average='macro', zero_division=0) * 100
        src_gate = test_alphas[mask].mean()
        print(f"  • {src:15s} | N: {mask.sum():5d} | Acc: {src_acc:.2f}% | F1: {src_f1:.2f}% | Avg Gate α: {src_gate:.3f}")
    print("=" * 65)


# ==============================================================================
# 5. CLI ENTRYPOINT
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pelatihan AIForensicNet V4 Universal pada Quad-Dataset (CASIA2, Defactify, Thunderbeast, IMD2020)")
    parser.add_argument('--epochs', type=int, default=20, help="Jumlah epoch pelatihan (default: 20)")
    parser.add_argument('--batch-size', type=int, default=32, help="Ukuran batch (default: 32)")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate awal (default: 5e-4)")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="Weight decay L2 regulasi")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout probability pada classifier head")
    parser.add_argument('--casia-dir', type=str, default='data/casia2', help="Direktori CASIA v2")
    parser.add_argument('--defactify-samples', type=int, default=6000, help="Jumlah sampel per kelas Defactify (default: 6000)")
    parser.add_argument('--thunderbeast-samples', type=int, default=6000, help="Jumlah sampel per kelas Thunderbeast (default: 6000)")
    parser.add_argument('--max-samples-per-class', type=int, default=15000, help="Maksimal citra seimbang per kelas final (default: 15000)")
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'ce'], help="Loss function: 'focal' (Focal Loss gamma=2) atau 'ce' (CrossEntropy)")
    parser.add_argument('--output-path', type=str, default='best_aiforensicnet_v4_quad.pth', help="Nama file checkpoint bobot model")
    parser.add_argument('--universal-backup', action='store_true', default=True, help="Simpan salinan ke best_aiforensicnet_v4_universal.pth")
    parser.add_argument('--amp', action='store_true', default=True, help="Gunakan FP16 Automatic Mixed Precision")
    parser.add_argument('--device', type=str, default='cuda', help="Device pelatihan (cuda atau cpu)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed reproduksibilitas")

    args = parser.parse_args()
    train_quad_model(args)
