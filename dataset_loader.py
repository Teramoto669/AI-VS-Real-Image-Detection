import warnings
warnings.filterwarnings('ignore', message='.*Palette images with Transparency.*')
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class KaggleAIVsHumanDataset(Dataset):
    """
    Dataset loader ultra-cepat untuk dataset Kaggle (alessandrasala79/ai-vs-human-generated-dataset).
    
    Fitur Optimasi Performa:
    1. Pre-resolusi path folder gambar 1x di __init__ (menghilangkan ratusan disk lookup per batch).
    2. Konversi metadata ke Python list murni (menghilangkan overhead pandas .iloc).
    3. Pre-komputasi penyelarasan label:
       - Kaggle Asli: 0 = Human, 1 = AI
       - Model Proyek: 0 = AI-Generated, 1 = Real Image
    """
    def __init__(self, df, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        img_col = 'file_name' if 'file_name' in df.columns else ('id' if 'id' in df.columns else df.columns[0])
        lbl_col = 'label' if 'label' in df.columns else df.columns[1]

        # 1. Cari folder gambar utama satu kali saja saat inisialisasi
        self.img_dir = data_dir
        for sub in ['train_data', 'train_data_v2', 'test_data', 'test_data_v2']:
            sub_path = os.path.join(data_dir, sub)
            if os.path.exists(sub_path) and os.path.isdir(sub_path):
                self.img_dir = sub_path
                break

        # 2. Simpan nama file dan label sebagai list murni (sangat cepat di __getitem__)
        self.filenames = [os.path.basename(str(f)) for f in df[img_col]]
        raw_labels = df[lbl_col].tolist()
        
        # Penyelarasan label: Kaggle 1 (AI) -> 0, Kaggle 0 (Human) -> 1
        self.labels = [0 if r == 1 else 1 for r in raw_labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Fallback cepat jika tidak ada di subfolder utama
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_dir, img_name)

        image = Image.open(img_path)
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]
