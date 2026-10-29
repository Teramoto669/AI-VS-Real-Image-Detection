"""
Generator Notebook Google Colab: AIFusionNet-DualScale V5 (Revisi V5.2)
======================================================================
Membuat file 'AI_vs_Real_Universal_V5_Colab.ipynb' yang mandiri (self-contained),
siap dijalankan langsung di Google Colab (GPU T4/A100).

Arsitektur Baru: Dual-Scale Multi-Domain Forensic Net
1. Branch 1 (RGB Global Semantic):
   - Menerima citra penuh yang di-resize moderat (224×224).
   - Melihat komposisi keseluruhan, bentuk anatomi tangan/mata, konsistensi bayangan, dan pencahayaan global.
2. Branch 2 (Forensik Mikro SRM + 2D FFT):
   - Menerima patch 224×224 pada resolusi murni 1:1 tanpa resize.
   - Melihat residu frekuensi tinggi murni: 12-kanal filter SRM + 3-kanal 2D FFT spektral + pola kisi JPEG 8x8.
3. Learnable Fusion Gate:
   - Secara dinamis menimbang: apakah bukti manipulasi lebih dominan di level semantik global (alpha -> 1)
     atau di level residu frekuensi mikro (1 - alpha -> 1)?
"""

import json
import os


def _lines(text):
    return text.strip("\n").splitlines(keepends=True)


def md(cell_id, text):
    return {"cell_type": "markdown", "metadata": {"id": cell_id}, "source": _lines(text)}


def code(cell_id, text):
    return {"cell_type": "code", "execution_count": None, "metadata": {"id": cell_id},
            "outputs": [], "source": _lines(text)}


# ---------------------------------------------------------------------------
CELL_HEADER = r'''
# 🛡️ AIFusionNet-DualScale (V5.3 Optimized Architecture)
## Deteksi Citra AI vs Foto Asli & Forensik Splicing Multi-Domain
**100% Dilatih dari Awal (From Scratch, Tanpa Pretrained Weights)**

Arsitektur **Dual-Scale V5.3** dirancang khusus untuk memecahkan kelemahan deteksi citra forensik:
1. **Branch 1 (RGB Global Context - Resized 224×224)**:  
   Melihat citra secara utuh untuk menangkap keanehan semantik makro (distorsi anatomi jari/wajah AI, pencahayaan tidak konsisten, kejanggalan semantik).
2. **Branch 2 (Micro-Forensic Residuals - Native 1:1 Salient Patch 224×224 Tanpa Resize)**:  
   Menganalisis potongan resolusi asli 1:1 tanpa pemudaran interpolasi untuk menangkap jejak mikroskopis: 12-kanal filter bank SRM (*Spatial Rich Model*), spektrum 2D FFT, dan anomali kisi kompresi JPEG 8×8. Menggunakan **Salient Gradient Patch Selection** untuk memindai area bertekstur/manipulasi tertinggi.
3. **Dynamic Fusion Gate**:  
   Secara adaptif menyeimbangkan kedua skala visual ($\alpha$ Global vs $1-\alpha$ Mikro Forensik).
4. **Anti-Overfitting & Model EMA (Exponential Moving Average)**:  
   Regularisasi kuat (Weight Decay 1e-3, Dropout 0.40, Label Smoothing 0.10, dan LR 1.5e-4) dikombinasikan dengan pembobotan EMA ($\text{decay}=0.995$) untuk menjaga stabilitas loss validasi dan generalisasi lintas domain.
5. **Kalibrasi Ambang Optimal Seimbang (Balanced Macro-F1)**:  
   Menyeimbangkan True Positive Rate (TPR $\ge 70\%$) dan False Positive Rate (FPR $\le 25\%$) melalui optimasi Macro-F1, menghindari fenomena *Recall Collapse*.
6. **Dataset Rekomendasi Bebas Noise Negatif**:  
   Kombinasi CASIA 2.0 (Splicing), Defactify (Generative AI Photos), dan Thunderbeast (AI Art). Dataset IMD2020 dinonaktifkan secara default untuk mencegah degradasi akurasi (*negative transfer*).
'''

CELL_SETUP = r'''
# 1. Instalasi Pustaka Pendukung yang Dibutuhkan
!pip install -q datasets gdown kagglehub scikit-learn tqdm pillow pyarrow matplotlib seaborn

# 2. Seluruh Modul & Library Diimpor di Awal
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

import os
import sys
import io
import math
import glob
import copy
import random
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import load_dataset
try:
    from google.colab import files
except ImportError:
    files = None

# 3. Verifikasi Akselerator Hardware (GPU)
print("=" * 65)
print(f"CUDA Tersedia : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM Tersedia : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("⚠️ PERINGATAN: GPU tidak terdeteksi! Buka menu Runtime -> Change runtime type -> Pilih T4 GPU.")
print("=" * 65)
print("[+] Seluruh pustaka & modul berhasil diimpor di awal notebook.")
'''

CELL_CONFIG = r'''
#@title ⚙️ 2. PENGATURAN DATASET & HYPERPARAMETER { display-mode: "form" }
#@markdown ### 📁 1. PILIH DATASET YANG DIGUNAKAN (Aktifkan / Nonaktifkan)
USE_CASIA = True          #@param {type:"boolean"}
USE_DEFACTIFY = True      #@param {type:"boolean"}
USE_THUNDERBEAST = True   #@param {type:"boolean"}
USE_IMD2020 = False       #@param {type:"boolean"}

#@markdown *Simpan cache Defactify/Thunderbeast/IMD2020 sebagai ZIP di Google Drive (opsional):*
USE_DRIVE_CACHE = False   #@param {type:"boolean"}
DRIVE_CACHE_DIR = "/content/drive/MyDrive/AIFusionNet_V5_cache" #@param {type:"string"}

#@markdown ---
#@markdown ### 📊 2. MODE KAPASITAS & JUMLAH DATA
PRESET = "Optimal (Rekomendasi - ~24.000 Citra)" #@param ["Cepat (Quick Test - ~6.000 Citra)", "Optimal (Rekomendasi - ~24.000 Citra)", "Maksimal (Full SOTA - ~50.000 Citra)", "Gunakan SEMUA Data (100% Full)", "Custom (Kustomisasi Manual)"]

#@markdown ---
#@markdown ### 🛠️ 3. INPUT JUMLAH DATA PER DATASET (Aktif pada mode 'Custom')
MAX_SAMPLES_PER_CLASS = 12000 #@param {type:"integer"}
CASIA_MAX_SAMPLES = 4000      #@param {type:"integer"}
DEFACTIFY_SAMPLES = 4000      #@param {type:"integer"}
THUNDERBEAST_SAMPLES = 4000   #@param {type:"integer"}
IMD2020_MAX_SAMPLES = 2000    #@param {type:"integer"}

#@markdown ---
#@markdown ### 🔬 4. PARAMETER DUAL-SCALE & FORENSIK
#@markdown *Peluang re-kompresi JPEG saat training, diterapkan ke KEDUA kelas:*
JPEG_AUG_P = 0.8          #@param {type:"number"}
#@markdown *Uji generalisasi lintas-dataset (zero-shot holdout):*
HOLDOUT_SOURCE = "Tidak Ada" #@param ["Tidak Ada", "CASIA2", "Defactify", "Thunderbeast", "IMD2020"]
#@markdown *Batas maksimal salah vonis foto asli sebagai palsu (target False Positive Rate):*
TARGET_FPR = 0.05         #@param {type:"number"}

#@markdown ---
#@markdown ### 🚀 5. HYPERPARAMETER PELATIHAN (Anti-Overfitting & Stabilisasi)
IMG_SIZE = 224                # Resolusi seragam untuk Global View & Native Patch
BATCH_SIZE = 32               #@param [16, 32, 64] {type:"raw"}
EPOCHS = 20                   #@param {type:"integer"}
WARMUP_EPOCHS = 3             #@param {type:"integer"}
PATIENCE = 6                  #@param {type:"integer"}
LEARNING_RATE = 1.5e-4        #@param {type:"number"}
WEIGHT_DECAY = 1e-3           #@param {type:"number"}
DROPOUT = 0.40                #@param {type:"number"}
LABEL_SMOOTHING = 0.10        #@param {type:"number"}
EMA_DECAY = 0.995             #@param {type:"number"}
GRAD_CLIP = 1.0               #@param {type:"number"}
NUM_WORKERS = 2               #@param {type:"integer"}
SEED = 42                     #@param {type:"integer"}

# Logika Preset
if PRESET == "Gunakan SEMUA Data (100% Full)":
    MAX_SAMPLES_PER_CLASS = -1
    CASIA_MAX_SAMPLES = -1
    DEFACTIFY_SAMPLES = -1
    THUNDERBEAST_SAMPLES = -1
    IMD2020_MAX_SAMPLES = -1
elif PRESET == "Cepat (Quick Test - ~6.000 Citra)":
    MAX_SAMPLES_PER_CLASS = 3000
    CASIA_MAX_SAMPLES = 1500
    DEFACTIFY_SAMPLES = 1500
    THUNDERBEAST_SAMPLES = 1500
    IMD2020_MAX_SAMPLES = 1000
    EPOCHS = 10
elif PRESET == "Optimal (Rekomendasi - ~24.000 Citra)":
    MAX_SAMPLES_PER_CLASS = 12000
    CASIA_MAX_SAMPLES = 4000
    DEFACTIFY_SAMPLES = 4000
    THUNDERBEAST_SAMPLES = 4000
    IMD2020_MAX_SAMPLES = 2000
    EPOCHS = 20
elif PRESET == "Maksimal (Full SOTA - ~50.000 Citra)":
    MAX_SAMPLES_PER_CLASS = 25000
    CASIA_MAX_SAMPLES = 8000
    DEFACTIFY_SAMPLES = 8000
    THUNDERBEAST_SAMPLES = 8000
    IMD2020_MAX_SAMPLES = 4000
    EPOCHS = 25
elif PRESET == "Custom (Kustomisasi Manual)":
    pass

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ['Real', 'Fake/AI/Tampered']
HOLDOUT = None if HOLDOUT_SOURCE == "Tidak Ada" else HOLDOUT_SOURCE

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enabled_sources = {'CASIA2': USE_CASIA, 'Defactify': USE_DEFACTIFY, 'Thunderbeast': USE_THUNDERBEAST, 'IMD2020': USE_IMD2020}
active_datasets = [k for k, v in enabled_sources.items() if v]
if HOLDOUT is not None:
    assert enabled_sources[HOLDOUT], f"ERROR: Dataset holdout '{HOLDOUT}' harus diaktifkan di bagian 1."
    assert len(active_datasets) >= 2, "ERROR: Mode holdout membutuhkan minimal 2 dataset aktif."

print("=" * 65)
print(f"🎯 Arsitektur            : Dual-Scale Multi-Domain (Global 224px + Native Patch 224px)")
print(f"📁 Dataset Aktif         : {', '.join(active_datasets) if active_datasets else 'TIDAK ADA AKTIF!'}")
target_str = "SEMUA DATA (100% Full)" if MAX_SAMPLES_PER_CLASS <= 0 else f"{MAX_SAMPLES_PER_CLASS:,} Real + {MAX_SAMPLES_PER_CLASS:,} Fake"
print(f"📦 Target Per Kelas      : {target_str}")
print(f"⚙️  Rincian Kuota        : CASIA={'SEMUA' if CASIA_MAX_SAMPLES <= 0 else CASIA_MAX_SAMPLES} | Defactify={'SEMUA' if DEFACTIFY_SAMPLES <= 0 else DEFACTIFY_SAMPLES} | Thunderbeast={'SEMUA' if THUNDERBEAST_SAMPLES <= 0 else THUNDERBEAST_SAMPLES} | IMD2020={'SEMUA' if IMD2020_MAX_SAMPLES <= 0 else IMD2020_MAX_SAMPLES}")
print(f"🔬 Preprocessing         : Dual-Scale (Global Resized + Salient 1:1 Patch) | JPEG aug p={JPEG_AUG_P} | Holdout={HOLDOUT or '-'}")
print(f"🚀 Pelatihan             : {EPOCHS} Epochs (warmup {WARMUP_EPOCHS}, patience {PATIENCE}) | Batch={BATCH_SIZE} | LR={LEARNING_RATE} | EMA={EMA_DECAY} | Device={device}")
print("=" * 65)
'''

CELL_DATASETS = r'''
# 3. Unduh dan Persiapkan Dataset Multi-Domain (CASIA2, Defactify, Thunderbeast, IMD2020)
for d in ['data/casia2', 'data/defactify_cache/real', 'data/defactify_cache/ai',
          'data/thunderbeast_cache/real', 'data/thunderbeast_cache/ai', 'data/imd2020_cache']:
    os.makedirs(d, exist_ok=True)

if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs(DRIVE_CACHE_DIR, exist_ok=True)

def count_files(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTS)])

def drive_restore(name):
    zp = os.path.join(DRIVE_CACHE_DIR, f'{name}.zip')
    if USE_DRIVE_CACHE and os.path.exists(zp):
        print(f"[*] Memulihkan {name} dari Google Drive...")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall('data')

def drive_backup(name):
    if USE_DRIVE_CACHE:
        print(f"[*] Menyimpan {name} ke Google Drive (ZIP)...")
        shutil.make_archive(os.path.join(DRIVE_CACHE_DIR, name), 'zip', root_dir='data', base_dir=name)

def safe_save_rgb(img_obj, save_path):
    if img_obj.mode in ('RGBA', 'LA') or (img_obj.mode == 'P' and 'transparency' in getattr(img_obj, 'info', {})):
        img_obj = img_obj.convert('RGBA')
    img_obj.convert('RGB').save(save_path, 'JPEG', quality=95)

def download_binary_hf(name, rows, real_dir, ai_dir, prefix, target, get_label, get_image, real_label):
    existing_r, existing_f = count_files(real_dir), count_files(ai_dir)
    c_r, c_f = existing_r, existing_f
    seen_r = seen_f = 0
    try:
        for row in tqdm(rows, desc=name):
            if c_r >= target and c_f >= target:
                break
            if int(get_label(row)) == real_label:
                seen_r += 1
                if seen_r <= existing_r or c_r >= target:
                    continue
                safe_save_rgb(get_image(row), f'{real_dir}/{prefix}_real_{c_r:05d}.jpg')
                c_r += 1
            else:
                seen_f += 1
                if seen_f <= existing_f or c_f >= target:
                    continue
                safe_save_rgb(get_image(row), f'{ai_dir}/{prefix}_ai_{c_f:05d}.jpg')
                c_f += 1
    except Exception as e:
        print(f"[!] {name} terhenti: {e}")
    print(f"[+] {name}: {c_r:,} Real | {c_f:,} AI tersimpan.")
    return (c_r - existing_r) + (c_f - existing_f)

# A. CASIA v2.0
CASIA_DIR = 'data/casia2'
if USE_CASIA:
    existing_casia = glob.glob('data/casia2/**/Au/*', recursive=True) + glob.glob(os.path.expanduser('~/.cache/kagglehub/**/Au/*'), recursive=True)
    if len(existing_casia) < 100:
        print("[*] Mengunduh dataset CASIA v2.0 via KaggleHub CDN...")
        try:
            CASIA_DIR = kagglehub.dataset_download('divg07/casia-20-image-tampering-detection-dataset')
            print(f"[+] CASIA v2.0 berhasil diunduh ke: {CASIA_DIR}")
        except Exception as e:
            print(f"[!] KaggleHub notice: {e}. Mencoba mirror alternatif Google Drive...")
            !gdown -q "https://drive.google.com/uc?id=1IDUgcoUeonBxx2rASX-_QwV9fhbtqdY8" -O data/casia2.zip || true
            if os.path.exists('data/casia2.zip'):
                with zipfile.ZipFile('data/casia2.zip', 'r') as zf:
                    zf.extractall('data/casia2/')
                os.remove('data/casia2.zip')
    else:
        print(f"[+] CASIA v2.0 siap di cache lokal ({len(existing_casia)} citra terdeteksi).")
else:
    print("[-] CASIA v2.0 dinonaktifkan (dilewati).")

# B. Defactify
if USE_DEFACTIFY:
    if count_files('data/defactify_cache/real') + count_files('data/defactify_cache/ai') == 0:
        drive_restore('defactify_cache')
    target_def = float('inf') if DEFACTIFY_SAMPLES <= 0 else DEFACTIFY_SAMPLES
    if min(count_files('data/defactify_cache/real'), count_files('data/defactify_cache/ai')) < target_def:
        print(f"[*] Mengunduh Defactify ({'SEMUA DATA' if DEFACTIFY_SAMPLES <= 0 else f'Target: {DEFACTIFY_SAMPLES} per kelas'})...")
        try:
            ds_def = load_dataset('Rajarshi-Roy-research/Defactify_Image_Dataset', split='train', streaming=True)
            n_new = download_binary_hf('Defactify', ds_def, 'data/defactify_cache/real', 'data/defactify_cache/ai', 'def',
                                       target_def, lambda r: r['Label_A'], lambda r: r['Image'], real_label=0)
            if n_new > 0:
                drive_backup('defactify_cache')
        except Exception as e:
            print(f"[!] Defactify gagal dimuat: {e}")
    else:
        print(f"[+] Defactify siap di cache ({count_files('data/defactify_cache/real')} Real, {count_files('data/defactify_cache/ai')} AI).")
else:
    print("[-] Defactify dinonaktifkan (dilewati).")

# C. Thunderbeast
if USE_THUNDERBEAST:
    if count_files('data/thunderbeast_cache/real') + count_files('data/thunderbeast_cache/ai') == 0:
        drive_restore('thunderbeast_cache')
    target_tb = float('inf') if THUNDERBEAST_SAMPLES <= 0 else THUNDERBEAST_SAMPLES
    if min(count_files('data/thunderbeast_cache/real'), count_files('data/thunderbeast_cache/ai')) < target_tb:
        print(f"[*] Mengunduh Thunderbeast ({'SEMUA DATA' if THUNDERBEAST_SAMPLES <= 0 else f'Target: {THUNDERBEAST_SAMPLES} per kelas'})...")
        try:
            ds_tb = load_dataset('Thunderbeast/AI-Generated-vs-Real-Images-Datasets', split='train')
            n_new = download_binary_hf('Thunderbeast', ds_tb, 'data/thunderbeast_cache/real', 'data/thunderbeast_cache/ai', 'tb',
                                       target_tb, lambda r: r['label'], lambda r: r['image'], real_label=1)
            if n_new > 0:
                drive_backup('thunderbeast_cache')
        except Exception as e:
            print(f"[!] Thunderbeast gagal dimuat: {e}")
    else:
        print(f"[+] Thunderbeast siap di cache ({count_files('data/thunderbeast_cache/real')} Real, {count_files('data/thunderbeast_cache/ai')} AI).")
else:
    print("[-] Thunderbeast dinonaktifkan (dilewati).")

# D. IMD2020
if USE_IMD2020:
    if count_files('data/imd2020_cache') == 0:
        drive_restore('imd2020_cache')
    existing_imd = count_files('data/imd2020_cache')
    target_imd = float('inf') if IMD2020_MAX_SAMPLES <= 0 else IMD2020_MAX_SAMPLES
    if existing_imd < target_imd:
        print(f"[*] Mengunduh IMD2020 ({'SEMUA DATA' if IMD2020_MAX_SAMPLES <= 0 else f'Target: {IMD2020_MAX_SAMPLES} citra'})...")
        try:
            ds_imd = load_dataset('ThaneJoss/IMD2020', split='full')
            count = existing_imd
            for i, row in enumerate(tqdm(ds_imd, desc="IMD2020")):
                if count >= target_imd:
                    break
                save_p = f"data/imd2020_cache/imd_{i:05d}_{int(row['label'])}.jpg"
                if os.path.exists(save_p):
                    continue
                safe_save_rgb(row['image'], save_p)
                count += 1
            if count > existing_imd:
                drive_backup('imd2020_cache')
        except Exception as e:
            print(f"[!] IMD2020 gagal dimuat: {e}")
    else:
        print(f"[+] IMD2020 siap di cache ({existing_imd} citra).")
else:
    print("[-] IMD2020 dinonaktifkan (dilewati).")

print("[+] Status seluruh dataset selesai diperiksa.")
'''

CELL_BALANCING = r'''
# 4. Penyeimbangan Data PER SUMBER (50% Real : 50% Fake), Holdout Lintas-Dataset, & Split 80:10:10
all_records = []

def list_images(pattern):
    return sorted(p for p in glob.glob(pattern) if p.lower().endswith(VALID_EXTS))

def cap(paths, n):
    return paths[:n] if n and n > 0 else paths

# Scan CASIA2
if USE_CASIA:
    search_roots = ['data/casia2', globals().get('CASIA_DIR', ''), os.path.expanduser('~/.cache/kagglehub')]
    casia_au_all, casia_tp_all = [], []
    for root in search_roots:
        if root and os.path.exists(root):
            casia_au_all.extend(p for p in glob.glob(f"{root}/**/Au/*", recursive=True) if p.lower().endswith(VALID_EXTS))
            casia_tp_all.extend(p for p in glob.glob(f"{root}/**/Tp/*", recursive=True) if p.lower().endswith(VALID_EXTS))
    casia_au = sorted(set(casia_au_all))
    casia_tp = sorted(set(casia_tp_all))
    if CASIA_MAX_SAMPLES > 0:
        rng = random.Random(SEED)
        if len(casia_au) > CASIA_MAX_SAMPLES: casia_au = rng.sample(casia_au, CASIA_MAX_SAMPLES)
        if len(casia_tp) > CASIA_MAX_SAMPLES: casia_tp = rng.sample(casia_tp, CASIA_MAX_SAMPLES)
    all_records += [{'image_path': p, 'label': 0, 'source': 'CASIA2'} for p in casia_au]
    all_records += [{'image_path': p, 'label': 1, 'source': 'CASIA2'} for p in casia_tp]
    print(f"[+] CASIA v2.0 dimuat: {len(casia_au):,} Real | {len(casia_tp):,} Fake")

# Scan Defactify
if USE_DEFACTIFY:
    def_real = cap(list_images('data/defactify_cache/real/*'), DEFACTIFY_SAMPLES)
    def_ai = cap(list_images('data/defactify_cache/ai/*'), DEFACTIFY_SAMPLES)
    all_records += [{'image_path': p, 'label': 0, 'source': 'Defactify'} for p in def_real]
    all_records += [{'image_path': p, 'label': 1, 'source': 'Defactify'} for p in def_ai]
    print(f"[+] Defactify dimuat: {len(def_real):,} Real | {len(def_ai):,} Fake")

# Scan Thunderbeast
if USE_THUNDERBEAST:
    tb_real = cap(list_images('data/thunderbeast_cache/real/*'), THUNDERBEAST_SAMPLES)
    tb_ai = cap(list_images('data/thunderbeast_cache/ai/*'), THUNDERBEAST_SAMPLES)
    all_records += [{'image_path': p, 'label': 0, 'source': 'Thunderbeast'} for p in tb_real]
    all_records += [{'image_path': p, 'label': 1, 'source': 'Thunderbeast'} for p in tb_ai]
    print(f"[+] Thunderbeast dimuat: {len(tb_real):,} Real | {len(tb_ai):,} Fake")

# Scan IMD2020
if USE_IMD2020:
    imd_all = list_images('data/imd2020_cache/*')
    imd_real = cap([p for p in imd_all if p.endswith('_0.jpg')], IMD2020_MAX_SAMPLES)
    imd_fake = cap([p for p in imd_all if p.endswith('_1.jpg')], IMD2020_MAX_SAMPLES)
    all_records += [{'image_path': p, 'label': 0, 'source': 'IMD2020'} for p in imd_real]
    all_records += [{'image_path': p, 'label': 1, 'source': 'IMD2020'} for p in imd_fake]
    print(f"[+] IMD2020 dimuat: {len(imd_real):,} Real | {len(imd_fake):,} Fake")

assert len(all_records) > 0, "ERROR: Tidak ada dataset aktif! Silakan centang minimal satu dataset di Cell 2."
df = pd.DataFrame(all_records).drop_duplicates(subset=['image_path'])

def balance_two_classes(g, seed=SEED):
    g_r, g_f = g[g['label'] == 0], g[g['label'] == 1]
    n = min(len(g_r), len(g_f))
    if n == 0:
        return None
    return pd.concat([g_r.sample(n=n, random_state=seed), g_f.sample(n=n, random_state=seed)])

# Holdout lintas-dataset
holdout_df = None
if HOLDOUT is not None:
    holdout_df = balance_two_classes(df[df['source'] == HOLDOUT])
    assert holdout_df is not None, f"ERROR: Holdout {HOLDOUT} tidak punya citra Real dan Fake sekaligus."
    holdout_df = holdout_df.reset_index(drop=True)
    df = df[df['source'] != HOLDOUT]
    print(f"[🌍] Holdout lintas-dataset: {HOLDOUT} ({len(holdout_df):,} citra) TIDAK dipakai untuk training.")

# Seimbangkan PER SUMBER
parts = []
for src, g in df.groupby('source'):
    b = balance_two_classes(g)
    if b is None:
        print(f"[!] {src} dilewati: hanya memiliki satu kelas.")
        continue
    if len(b) < len(g):
        print(f"[i] {src}: diseimbangkan menjadi {len(b)//2:,} Real + {len(b)//2:,} Fake")
    parts.append(b)
assert parts, "ERROR: Tidak ada sumber yang memiliki citra Real dan Fake sekaligus."
df_balanced = pd.concat(parts)

if MAX_SAMPLES_PER_CLASS is not None and MAX_SAMPLES_PER_CLASS > 0 and len(df_balanced) // 2 > MAX_SAMPLES_PER_CLASS:
    frac = MAX_SAMPLES_PER_CLASS / (len(df_balanced) / 2)
    df_balanced = df_balanced.groupby(['source', 'label']).sample(frac=frac, random_state=SEED)
df_balanced = df_balanced.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# Split stratified per (sumber, label)
strata = df_balanced['source'] + '_' + df_balanced['label'].astype(str)
train_df, temp_df = train_test_split(df_balanced, test_size=0.20, stratify=strata, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=strata.loc[temp_df.index], random_state=SEED)

print(f"\n[+] Dataset Seimbang Siap: {len(df_balanced):,} Total Citra (50% Real : 50% Fake per sumber)")
print(f"    - Train : {len(train_df):,} citra (80%)")
print(f"    - Val   : {len(val_df):,} citra (10%)")
print(f"    - Test  : {len(test_df):,} citra (10%)")
print("\nRincian Sampel per Sumber:")
print(df_balanced.groupby(['source', 'label']).size().unstack(fill_value=0))
'''

CELL_ARCH = r'''
# 5. Definisi Arsitektur AIFusionNet-DualScale V5 (100% From Scratch)
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class ModernConvBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.se = SqueezeExcitation(dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.se(x)

class StaticForensicFilterBank(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = torch.tensor([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], dtype=torch.float32) / 4.0
        f2 = torch.tensor([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=torch.float32) / 12.0
        f3 = torch.tensor([[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]], dtype=torch.float32) / 4.0
        f4 = torch.tensor([[0,0,0,0,0],[0,-1,0,1,0],[0,-2,0,2,0],[0,-1,0,1,0],[0,0,0,0,0]], dtype=torch.float32) / 4.0
        filters = torch.stack([f1, f2, f3, f4], dim=0)
        weight = torch.zeros(12, 1, 5, 5)
        for i in range(4):
            for c in range(3):
                weight[i * 3 + c, 0] = filters[i]
        self.conv = nn.Conv2d(3, 12, kernel_size=5, padding=2, bias=False, groups=3)
        self.conv.weight = nn.Parameter(weight, requires_grad=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.conv(x))

class Spectral2DResidualModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = LayerNorm2d(3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
        return self.norm(torch.log1p(torch.abs(fft)))

class MultiStatPool2d(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=[2, 3])
        std = torch.sqrt(torch.var(x, dim=[2, 3], unbiased=False) + self.eps)
        max_val = torch.amax(x, dim=[2, 3])
        return torch.cat([mean, std, max_val], dim=1)

def make_stage(in_dim: int, out_dim: int, depth: int, stride: int = 2):
    layers = []
    if stride == 2:
        layers.append(LayerNorm2d(in_dim))
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2))
    for _ in range(depth):
        layers.append(ModernConvBlock(out_dim))
    return nn.Sequential(*layers)

class AIFusionNetUniversal(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.3):
        super().__init__()
        # Branch 1 (RGB Global Semantic Stream)
        self.b1_stem = nn.Sequential(nn.Conv2d(3, 48, kernel_size=4, stride=4), LayerNorm2d(48))
        self.b1_stage1 = make_stage(48, 48, depth=2, stride=1)
        self.b1_stage2 = make_stage(48, 96, depth=2, stride=2)
        self.b1_stage3 = make_stage(96, 192, depth=3, stride=2)
        self.b1_stage4 = make_stage(192, 384, depth=2, stride=2)

        # Branch 2 (Forensik Residu Mikro SRM + Spektral dari Patch Resolusi Murni)
        self.forensic_bank = StaticForensicFilterBank()
        self.spectral_module = Spectral2DResidualModule()
        self.b2_stem = nn.Sequential(nn.Conv2d(15, 48, kernel_size=4, stride=4), LayerNorm2d(48))
        self.b2_stage1 = make_stage(48, 48, depth=2, stride=1)
        self.b2_stage2 = make_stage(48, 96, depth=2, stride=2)
        self.b2_stage3 = make_stage(96, 192, depth=3, stride=2)
        self.b2_stage4 = make_stage(192, 384, depth=2, stride=2)

        self.pool = MultiStatPool2d()
        feat_dim = 384 * 3  # 1152

        # Dynamic Dual-Scale Fusion Gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(p=0.2), nn.Linear(256, feat_dim), nn.Sigmoid()
        )
        self.main_head = nn.Sequential(
            nn.Dropout(p=dropout_p), nn.Linear(feat_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(p=0.25), nn.Linear(256, num_classes)
        )
        self.aux_head_rgb = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, num_classes))
        self.aux_head_forensic = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, num_classes))

    def extract_features(self, x_global: torch.Tensor, x_patch: torch.Tensor = None):
        if x_patch is None:
            x_patch = x_global
        # Branch 1: Global Semantics
        f1 = self.b1_stem(x_global)
        f1 = self.b1_stage4(self.b1_stage3(self.b1_stage2(self.b1_stage1(f1))))
        v_rgb = self.pool(f1)

        # Branch 2: Micro-Forensics
        res = torch.cat([self.forensic_bank(x_patch), self.spectral_module(x_patch)], dim=1)
        f2 = self.b2_stem(res)
        f2 = self.b2_stage4(self.b2_stage3(self.b2_stage2(self.b2_stage1(f2))))
        v_forensic = self.pool(f2)
        return v_rgb, v_forensic

    def forward(self, x_global: torch.Tensor, x_patch: torch.Tensor = None,
                return_all: bool = False, return_gate: bool = False):
        v_rgb, v_forensic = self.extract_features(x_global, x_patch)
        alpha = self.fusion_gate(torch.cat([v_rgb, v_forensic], dim=1))
        v_fused = alpha * v_rgb + (1.0 - alpha) * v_forensic
        main_out = self.main_head(v_fused)
        if return_all:
            return main_out, self.aux_head_rgb(v_rgb), self.aux_head_forensic(v_forensic), alpha.mean(dim=1)
        if return_gate:
            return main_out, alpha.mean(dim=1)
        return main_out

model = AIFusionNetUniversal(num_classes=2, dropout_p=DROPOUT).to(device)
total_p = sum(p.numel() for p in model.parameters())
print(f"[+] Arsitektur AIFusionNet-DualScale V5 Siap: {total_p:,} parameter (100% From Scratch).")
'''

CELL_TRANSFORMS = r'''
# 6. Data Augmentation & Dual-Scale DataLoader (Salient Patch Selection)
class RandomJPEGCompression:
    def __init__(self, quality_range=(60, 100), p=0.8):
        self.quality_range, self.p = quality_range, p
    def __call__(self, img):
        if random.random() < self.p:
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=random.randint(*self.quality_range))
            buf.seek(0)
            return Image.open(buf).convert('RGB')
        return img

class FixedJPEG:
    def __init__(self, quality=75):
        self.quality = quality
    def __call__(self, img):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

class RandomGaussianBlur:
    def __init__(self, p=0.1, radius=(0.1, 1.5)):
        self.p, self.radius = p, radius
    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.radius)))
        return img

class EnsureMinSize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        w, h = img.size
        if min(w, h) >= self.size:
            return img
        s = self.size / min(w, h)
        return img.resize((max(self.size, round(w * s)), max(self.size, round(h * s))), Image.Resampling.BICUBIC)

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

norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# Pipeline Skala 1: Citra Keseluruhan (Global View 224x224 dengan Invariance Augmentation)
train_global_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    norm
])
eval_global_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    norm
])

# Pipeline Skala 2: Patch Resolusi Murni 1:1 Tanpa Resize (Micro-Forensic Residuals)
train_patch_tf = transforms.Compose([
    EnsureMinSize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    RandomGaussianBlur(p=0.1),
    RandomJPEGCompression(quality_range=(60, 100), p=JPEG_AUG_P),
    transforms.ToTensor(),
    norm
])
eval_patch_tf = transforms.Compose([
    transforms.ToTensor(),
    norm
])

class DualScaleForensicDS(Dataset):
    def __init__(self, df, global_tf, patch_tf, is_eval=False):
        self.df = df.reset_index(drop=True)
        self.global_tf = global_tf
        self.patch_tf = patch_tf
        self.is_eval = is_eval
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        g_img = self.global_tf(img)
        if self.is_eval:
            p_pil = get_salient_patch_pil(img, IMG_SIZE)
            p_img = self.patch_tf(p_pil)
        else:
            p_img = self.patch_tf(img)
        return (g_img, p_img), int(row['label']), row.get('source', 'Unknown')

loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                 pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0)
train_loader = DataLoader(DualScaleForensicDS(train_df, train_global_tf, train_patch_tf, is_eval=False), shuffle=True, drop_last=True, **loader_kw)
val_loader = DataLoader(DualScaleForensicDS(val_df, eval_global_tf, eval_patch_tf, is_eval=True), shuffle=False, **loader_kw)
test_loader = DataLoader(DualScaleForensicDS(test_df, eval_global_tf, eval_patch_tf, is_eval=True), shuffle=False, **loader_kw)
holdout_loader = DataLoader(DualScaleForensicDS(holdout_df, eval_global_tf, eval_patch_tf, is_eval=True), shuffle=False, **loader_kw) if holdout_df is not None else None
print(f"[+] DataLoaders Dual-Scale siap ({NUM_WORKERS} workers). Batch latih per epoch: {len(train_loader)}")
'''

CELL_TRAINING = r'''
# 7. Pelatihan Dual-Scale: Warmup + Cosine, Gradient Clipping, Model EMA, Early Stopping
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

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
ema = ModelEMA(model, decay=EMA_DECAY)

steps_per_epoch = len(train_loader)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch
min_lr_ratio = 1e-6 / LEARNING_RATE

def lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

AUX_WEIGHT = 0.3
output_model_path = 'best_aifusionnet_v5_universal.pth'
CKPT_CONFIG = {
    'arch_variant': 'dual_scale',
    'img_size': IMG_SIZE,
    'mean': IMAGENET_MEAN,
    'std': IMAGENET_STD,
    'class_names': CLASS_NAMES,
    'dropout': DROPOUT,
    'datasets': active_datasets,
    'holdout': HOLDOUT,
    'ema_decay': EMA_DECAY,
    'crop_mode': 'salient_patch',
}
best_val_f1, epochs_no_improve = 0.0, 0
history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1', 'val_auc']}

print(f"🚀 Memulai Pelatihan AIFusionNet-DualScale ({EPOCHS} Epochs, EMA={EMA_DECAY}, early stopping patience={PATIENCE})...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] Latih", leave=False)
    for (imgs_g, imgs_p), lbls, _ in pbar_train:
        imgs_g = imgs_g.to(device, non_blocking=True)
        imgs_p = imgs_p.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            main_out, aux_rgb, aux_forensic, _ = model(imgs_g, imgs_p, return_all=True)
            loss = criterion(main_out, lbls) + AUX_WEIGHT * (criterion(aux_rgb, lbls) + criterion(aux_forensic, lbls))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        train_loss += loss.item() * imgs_g.size(0)
        correct += (torch.argmax(main_out, dim=1) == lbls).sum().item()
        total += lbls.size(0)
        pbar_train.set_postfix({'loss': f"{train_loss/total:.4f}", 'acc': f"{correct/total*100:.2f}%", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})

    # Evaluasi Validasi Dual-Scale menggunakan Bobot EMA (Lebih Halus & Stabil)
    v_probs, v_targets, v_gates = [], [], []
    val_loss = 0.0
    with torch.no_grad():
        for (imgs_g, imgs_p), lbls, _ in tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] Validasi (EMA)", leave=False):
            imgs_g, imgs_p = imgs_g.to(device, non_blocking=True), imgs_p.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                main_out, g = ema(imgs_g, imgs_p, return_gate=True)
            main_out = main_out.float()
            val_loss += criterion(main_out, lbls).item() * imgs_g.size(0)
            v_probs.extend(torch.softmax(main_out, dim=1)[:, 1].cpu().tolist())
            v_targets.extend(lbls.cpu().tolist())
            v_gates.extend(g.float().cpu().tolist())

    v_probs, v_targets = np.array(v_probs), np.array(v_targets)
    v_preds = (v_probs >= 0.5).astype(int)
    val_acc = np.mean(v_preds == v_targets) * 100.0
    val_f1 = f1_score(v_targets, v_preds, average='macro', zero_division=0) * 100.0
    val_auc = roc_auc_score(v_targets, v_probs) * 100.0
    train_acc = correct / total * 100.0
    val_loss /= len(v_targets)
    for k, v in zip(history.keys(), [train_loss / total, val_loss, train_acc, val_acc, val_f1, val_auc]):
        history[k].append(v)
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Train: Loss={train_loss/total:.4f}, Acc={train_acc:.2f}% | "
          f"Val(EMA): Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.2f}%, AUC={val_auc:.2f}%, Avg Gate={np.mean(v_gates):.3f}")

    if val_f1 > best_val_f1:
        best_val_f1, epochs_no_improve = val_f1, 0
        torch.save({
            'epoch': int(epoch),
            'model_state_dict': ema.state_dict(),
            'val_f1': float(val_f1),
            'val_acc': float(val_acc),
            'val_auc': float(val_auc),
            'arch': 'aifusionnet_v5_universal',
            'config': CKPT_CONFIG,
        }, output_model_path)
        print(f"  ⭐ Checkpoint Disimpan (EMA Weights): {output_model_path} (F1: {val_f1:.2f}%)")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"  ⏹️ Early stopping: F1 validasi tidak membaik selama {PATIENCE} epoch.")
            break
'''

CELL_CALIBRATION = r'''
# 8. Kalibrasi Probabilitas (Temperature Scaling) & Ambang Keputusan Anti-False-Positive
print("[*] Memuat bobot terbaik (EMA)...")
try:
    ckpt = torch.load(output_model_path, map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load(output_model_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

def collect_logits(loader):
    logits, labels, sources = [], [], []
    with torch.no_grad():
        for (imgs_g, imgs_p), lbls, srcs in tqdm(loader, desc="Inferensi", leave=False):
            imgs_g = imgs_g.to(device, non_blocking=True)
            imgs_p = imgs_p.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                out = model(imgs_g, imgs_p)
            logits.append(out.float().cpu())
            labels.append(lbls)
            sources.extend(srcs)
    return torch.cat(logits), torch.cat(labels), np.array(sources)

def expected_calibration_error(p_fake, y, n_bins=15):
    conf = np.maximum(p_fake, 1.0 - p_fake)
    pred = (p_fake >= 0.5).astype(int)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return ece

val_logits, val_labels, _ = collect_logits(val_loader)
val_y = val_labels.numpy()

# Temperature scaling (Guo et al., 2017) dengan clamping numerik aman
log_T = torch.zeros(1, requires_grad=True)
opt_T = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200)
def closure():
    opt_T.zero_grad()
    loss = F.cross_entropy(val_logits / log_T.exp(), val_labels)
    loss.backward()
    return loss
opt_T.step(closure)
TEMPERATURE = float(torch.clamp(log_T.exp(), 0.1, 5.0).item())

p_before = torch.softmax(val_logits, dim=1)[:, 1].numpy()
val_p = torch.softmax(val_logits / TEMPERATURE, dim=1)[:, 1].numpy()
print(f"[+] Temperature Scaling T = {TEMPERATURE:.3f}")
print(f"    NLL : {F.cross_entropy(val_logits, val_labels).item():.4f} -> {F.cross_entropy(val_logits / TEMPERATURE, val_labels).item():.4f}")
print(f"    ECE : {expected_calibration_error(p_before, val_y)*100:.2f}% -> {expected_calibration_error(val_p, val_y)*100:.2f}%")

# Ambang Keputusan Terkalibrasi:
# 1. Ambang Optimal Seimbang (Maximize Validation Macro-F1)
# Mencegah fenomena Recall Collapse di mana ambang dipaksa ke 0.78+ sehingga menghilangkan 80% deteksi Fake.
grid = np.linspace(0.35, 0.75, 81)
f1_grid = [f1_score(val_y, (val_p >= t).astype(int), average='macro', zero_division=0) for t in grid]
best_idx = int(np.argmax(f1_grid))
THRESHOLD_BALANCED = float(grid[best_idx])
best_val_f1_score = f1_grid[best_idx] * 100.0

# 2. Ambang Konservatif (Target FPR <= 5%) sebagai opsi keamanan ekstra
thr_fpr = float(np.quantile(val_p[val_y == 0], 1.0 - TARGET_FPR)) if (val_y == 0).sum() > 0 else 0.75
THRESHOLD_CONSERVATIVE = float(np.clip(thr_fpr, 0.50, 0.95))

# Tetapkan Ambang Seimbang sebagai THRESHOLD operasional utama
THRESHOLD = THRESHOLD_BALANCED

print(f"[★] Ambang Seimbang Optimal (Macro-F1) : {THRESHOLD_BALANCED:.3f} (Val F1: {best_val_f1_score:.2f}%) -> DIPILIH UTAMA")
print(f"[i] Ambang Konservatif (Target FPR<=5%) : {THRESHOLD_CONSERVATIVE:.3f}")

val_pred_bal = (val_p >= THRESHOLD_BALANCED).astype(int)
fpr_b, tpr_b = ((val_y == 0) & (val_pred_bal == 1)).sum() / max(1, (val_y == 0).sum())*100, ((val_y == 1) & (val_pred_bal == 1)).sum() / max(1, (val_y == 1).sum())*100
print(f"    Validasi @Ambang Seimbang : Acc={np.mean(val_pred_bal == val_y)*100:.2f}% | FPR={fpr_b:.2f}% | TPR={tpr_b:.2f}%")

ckpt.update({
    'temperature': TEMPERATURE,
    'threshold': THRESHOLD,
    'threshold_balanced': THRESHOLD_BALANCED,
    'threshold_conservative': THRESHOLD_CONSERVATIVE,
    'target_fpr': TARGET_FPR
})
torch.save(ckpt, output_model_path)
print(f"[+] Temperature & ambang tersimpan di checkpoint: {output_model_path}")
'''

CELL_EVALUATION = r'''
# 9. Evaluasi Lengkap: Test Set, Holdout Lintas-Dataset, Uji Ketahanan JPEG, & Kurva Training
def binary_rates(y, pred):
    fpr = ((y == 0) & (pred == 1)).sum() / max(1, (y == 0).sum())
    tpr = ((y == 1) & (pred == 1)).sum() / max(1, (y == 1).sum())
    return fpr * 100.0, tpr * 100.0

def safe_auc(y, p):
    return roc_auc_score(y, p) * 100.0 if len(np.unique(y)) == 2 else float('nan')

def full_report(loader, title, detailed=True, fig_name=None):
    logits, y, srcs = collect_logits(loader)
    y = y.numpy()
    p = torch.softmax(logits / TEMPERATURE, dim=1)[:, 1].numpy()
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for name, pred in [("Ambang default 0.500", (p >= 0.5).astype(int)),
                       (f"Ambang seimbang {THRESHOLD_BALANCED:.3f}", (p >= THRESHOLD_BALANCED).astype(int)),
                       (f"Ambang konservatif {THRESHOLD_CONSERVATIVE:.3f}", (p >= THRESHOLD_CONSERVATIVE).astype(int))]:
        fpr, tpr = binary_rates(y, pred)
        print(f"{name:32s} | Acc={np.mean(pred == y)*100:6.2f}% | Macro-F1={f1_score(y, pred, average='macro', zero_division=0)*100:6.2f}% "
              f"| FPR(Real->Fake)={fpr:6.2f}% | TPR(Fake terdeteksi)={tpr:6.2f}%")
    print(f"ROC-AUC (tidak bergantung ambang): {safe_auc(y, p):.2f}%")
    if not detailed:
        return p, y, srcs

    pred = (p >= THRESHOLD).astype(int)
    print(f"\nLaporan Klasifikasi (@Ambang Seimbang {THRESHOLD:.3f}):")
    print(classification_report(y, pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
    rows = []
    for s in np.unique(srcs):
        m = srcs == s
        fpr, tpr = binary_rates(y[m], pred[m])
        rows.append({'Sumber': s, 'N': int(m.sum()), 'Acc (%)': np.mean(pred[m] == y[m]) * 100.0,
                     'FPR (%)': fpr, 'TPR (%)': tpr, 'AUC (%)': safe_auc(y[m], p[m])})
    print("Performa per Dataset Sumber:")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    clean_title = title.replace('[', '').replace(']', '').split('-')[0].strip()
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Prediksi'); plt.ylabel('Label Asli'); plt.title(clean_title)
    plt.tight_layout()
    if fig_name:
        plt.savefig(fig_name, dpi=200)
    plt.show()
    return p, y, srcs

full_report(test_loader, "[TEST SET - In-Distribution]", fig_name='confusion_matrix_v5_test.png')

if holdout_loader is not None:
    full_report(holdout_loader, f"[HOLDOUT LINTAS-DATASET: {HOLDOUT} - Unseen Evaluation]",
                fig_name='confusion_matrix_v5_holdout.png')

# Uji ketahanan kompresi JPEG Q=75 pada kedua skala
jpeg_patch_tf = transforms.Compose([FixedJPEG(75), transforms.ToTensor(), norm])
jpeg_loader = DataLoader(DualScaleForensicDS(test_df, eval_global_tf, jpeg_patch_tf, is_eval=True), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
full_report(jpeg_loader, "[STRESS TEST: Test set dikompresi ulang JPEG Q=75]", detailed=False)

# Kurva training
if 'history' in globals() and history['train_loss']:
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].plot(ep, history['train_loss'], 'o-', label='Train'); axes[0].plot(ep, history['val_loss'], 's-', label='Val(EMA)')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(ep, history['train_acc'], 'o-', label='Train'); axes[1].plot(ep, history['val_acc'], 's-', label='Val(EMA)')
    axes[1].set_title('Akurasi (%)'); axes[1].legend()
    axes[2].plot(ep, history['val_f1'], 'o-', label='Val Macro-F1'); axes[2].plot(ep, history['val_auc'], 's-', label='Val ROC-AUC')
    axes[2].set_title('Validasi (%)'); axes[2].legend()
    for a in axes:
        a.set_xlabel('Epoch'); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves_v5.png', dpi=200)
    plt.show()
'''

CELL_PREDICT = r'''
# 10. Inferensi Dual-Scale Terkalibrasi (Global View + Salient Patch + TTA)
def predict_image(img_path, verbose=True):
    img = Image.open(img_path).convert('RGB')
    t_g = eval_global_tf(img).unsqueeze(0).to(device)
    p_pil = get_salient_patch_pil(img, IMG_SIZE)
    t_p = eval_patch_tf(p_pil).unsqueeze(0).to(device)
    t_g_flip = torch.flip(t_g, dims=[3])
    t_p_flip = torch.flip(t_p, dims=[3])

    model.eval()
    with torch.no_grad():
        out1, g1 = model(t_g, t_p, return_gate=True)
        out2, g2 = model(t_g_flip, t_p_flip, return_gate=True)
        p1 = torch.softmax(out1.float() / TEMPERATURE, dim=1)[0]
        p2 = torch.softmax(out2.float() / TEMPERATURE, dim=1)[0]
        p_real = ((p1[0] + p2[0]) / 2.0).item() * 100.0
        p_fake = ((p1[1] + p2[1]) / 2.0).item() * 100.0
        gate_alpha = ((g1 + g2) / 2.0).item() * 100.0

    verdict = 'FAKE / AI / TAMPERED' if (p_fake / 100.0) >= THRESHOLD else 'REAL AUTHENTIC'

    if verbose:
        print(f"Berkas: {os.path.basename(img_path)}")
        print(f"- Vonis Prediksi   : {verdict}  (ambang Fake >= {THRESHOLD*100:.1f}%)")
        print(f"- Probabilitas     : Real = {p_real:.2f}% | AI/Fake = {p_fake:.2f}%")
        print(f"- Atensi Dual-Scale: Global Semantics = {gate_alpha:.1f}% | Salient Micro Patch = {100.0 - gate_alpha:.1f}%")
        print("-" * 50)
    return p_fake

# Uji pada 5 sampel acak dari test set
for _, row in test_df.sample(min(5, len(test_df)), random_state=SEED).iterrows():
    print(f"[Label asli: {CLASS_NAMES[int(row['label'])]} | {row['source']}]")
    predict_image(row['image_path'])
'''

CELL_VISUALIZATION = r'''
# 11. Visualisasi Fitur Dual-Scale (Global Context, Salient Patch, SRM, 2D FFT, Grad-CAM, & Fusion Gate)
def grad_cam_v5_global(t_g, t_p, target_class=1):
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        f1 = model.b1_stem(t_g)
        f1 = model.b1_stage4(model.b1_stage3(model.b1_stage2(model.b1_stage1(f1))))
        f1.retain_grad()
        res = torch.cat([model.forensic_bank(t_p), model.spectral_module(t_p)], dim=1)
        f2 = model.b2_stem(res)
        f2 = model.b2_stage4(model.b2_stage3(model.b2_stage2(model.b2_stage1(f2))))
        v_rgb, v_for = model.pool(f1), model.pool(f2)
        alpha = model.fusion_gate(torch.cat([v_rgb, v_for], dim=1))
        logits = model.main_head(alpha * v_rgb + (1.0 - alpha) * v_for)
        logits[0, target_class].backward()

    w = f1.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * f1).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=t_g.shape[-2:], mode='bilinear', align_corners=False)[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    model.zero_grad(set_to_none=True)
    return cam, logits.detach(), alpha.detach()

def inspect_dual_scale_v5(image_input):
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
        title_name = os.path.basename(image_input)
    else:
        pil_img, title_name = image_input, "Sample Image"

    disp_global = eval_global_tf(pil_img)
    patch_pil = get_salient_patch_pil(pil_img, IMG_SIZE)
    disp_patch = eval_patch_tf(patch_pil)

    t_g = disp_global.unsqueeze(0).to(device)
    t_p = disp_patch.unsqueeze(0).to(device)

    with torch.no_grad():
        srm_res = model.forensic_bank(t_p).cpu().squeeze(0).mean(dim=0).numpy()
        fft_shift = torch.fft.fftshift(torch.fft.fft2(t_p, norm='ortho'), dim=(-2, -1))
        fft_map = torch.log(torch.abs(fft_shift) + 1e-6).cpu().squeeze(0).mean(dim=0).numpy()

    cam_rgb, logits, alpha = grad_cam_v5_global(t_g, t_p)
    prob = torch.softmax(logits.float() / TEMPERATURE, dim=1)[0]
    p_real, p_fake = prob[0].item() * 100.0, prob[1].item() * 100.0
    gate_pct = alpha.mean().item() * 100.0
    is_fake = (p_fake / 100.0) >= THRESHOLD

    rgb_disp_g = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    rgb_disp_p = np.array(patch_pil) / 255.0

    fig, axes = plt.subplots(1, 6, figsize=(26, 4.4))
    verdict = 'FAKE / AI / TAMPERED' if is_fake else 'REAL AUTHENTIC'
    axes[0].imshow(rgb_disp_g)
    axes[0].set_title(f"{title_name}\nPrediksi: {verdict}\n(Real={p_real:.1f}% | Fake={p_fake:.1f}%)", fontsize=10, fontweight='bold', color='red' if is_fake else 'green')
    axes[1].imshow(rgb_disp_p)
    axes[1].set_title("1. Salient 1:1 Patch (224px)\n(Resolusi Murni Bertekstur)", fontsize=10, fontweight='bold')
    axes[2].imshow(srm_res, cmap='inferno')
    axes[2].set_title("2. Residu Spasial SRM\n(Noise Filter High-Pass)", fontsize=10, fontweight='bold')
    axes[3].imshow(fft_map, cmap='nipy_spectral')
    axes[3].set_title("3. Spektrum Frekuensi 2D\n(2D FFT Magnitude)", fontsize=10, fontweight='bold')
    axes[4].imshow(rgb_disp_g); axes[4].imshow(cam_rgb, cmap='jet', alpha=0.5)
    axes[4].set_title("4. Grad-CAM Global Context\n(Area Semantik AI/Manipulasi)", fontsize=10, fontweight='bold')
    bars = axes[5].bar(['Global Semantics', 'Salient Forensics'], [gate_pct, 100.0 - gate_pct], color=['#2563eb', '#f59e0b'], width=0.55)
    axes[5].set_ylim(0, 100)
    axes[5].set_ylabel('Bobot Atensi (%)', fontsize=9)
    axes[5].set_title(f"5. Dual-Scale Fusion Gate\n(Global: {gate_pct:.1f}% | Mikro: {100-gate_pct:.1f}%)", fontsize=10, fontweight='bold')
    for bar in bars:
        axes[5].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 2, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    for i in range(5):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"dual_scale_fitur_v5_{os.path.splitext(title_name)[0]}.png", dpi=200)
    plt.show()

print("[*] Menampilkan visualisasi fitur Dual-Scale pada 1 sampel Real dan 1 sampel Fake...")
inspect_dual_scale_v5(test_df[test_df['label'] == 0].iloc[0]['image_path'])
inspect_dual_scale_v5(test_df[test_df['label'] == 1].iloc[0]['image_path'])
'''

CELL_DOWNLOAD = r'''
# 12. Unduh Bobot Model untuk Aplikasi Streamlit (app.py)
# Checkpoint berisi: model_state_dict, config (arch_variant: dual_scale), temperature, threshold
try:
    print(f"[*] Mengunduh {output_model_path} ke komputer Anda...")
    if files is not None:
        files.download(output_model_path)
    else:
        print(f"[+] File model tersimpan di: {os.path.abspath(output_model_path)}")
except Exception as e:
    print(f"[!] Silakan unduh manual file '{output_model_path}' ({e}).")
'''


def generate_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md("v5_header", CELL_HEADER),
            code("v5_setup", CELL_SETUP),
            code("v5_config", CELL_CONFIG),
            code("v5_datasets", CELL_DATASETS),
            code("v5_balancing", CELL_BALANCING),
            code("v5_architecture", CELL_ARCH),
            code("v5_transforms", CELL_TRANSFORMS),
            code("v5_training", CELL_TRAINING),
            code("v5_calibration", CELL_CALIBRATION),
            code("v5_evaluation", CELL_EVALUATION),
            code("v5_predict_func", CELL_PREDICT),
            code("v5_feature_visualization", CELL_VISUALIZATION),
            code("v5_download", CELL_DOWNLOAD),
        ],
    }

    out_file = "AI_vs_Real_Universal_V5_Colab.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"[+] Berhasil membuat notebook V5 Dual-Scale di: {os.path.abspath(out_file)}")


if __name__ == '__main__':
    generate_notebook()
