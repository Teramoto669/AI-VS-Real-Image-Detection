# AI vs Real Image Detection 🖼️🤖

Proyek Deep Learning berbasis **PyTorch** dan arsitektur **EfficientNet-B0** untuk mendeteksi dan mengklasifikasikan apakah suatu gambar merupakan buatan kecerdasan buatan (**AI-Generated**) atau foto/gambar asli (**Real**).

---

## 📌 Fitur Utama

- **Transfer Learning State-of-the-Art**: Menggunakan backbone pretrained **EfficientNet-B0** yang ringan, cepat, dan memiliki akurasi tinggi.
- **Dukungan Akselerasi GPU NVIDIA (CUDA)**: Teroptimasi untuk berjalan di GPU (misal: NVIDIA GeForce RTX 4050 Laptop GPU) menggunakan CUDA 12.6.
- **Integrasi Hugging Face Datasets**: Pemuatan dan pemrosesan dataset langsung dari Hugging Face Hub dengan format streaming/caching otomatis.
- **Augmentasi Data Dinamis**: Resizing ke resolusi standar (224x224), *Random Horizontal Flip*, dan *Random Rotation* untuk mencegah overfitting.
- **Checkpoint Otomatis**: Menyimpan bobot model terbaik (`best_efficientnet_b0_ai_vs_real.pth`) berdasarkan nilai akurasi validasi tertinggi.
- **Fungsi Inferensi Cepat**: Fungsi `predict_image()` bawaan untuk menguji gambar baru dengan output probabilitas keyakinan (*confidence score*).

---

## 📂 Struktur Proyek

```text
AI VS REAL/
├── AI vs Real image detection.ipynb    # Notebook utama pelatihan & evaluasi model
├── best_efficientnet_b0_ai_vs_real.pth # Bobot model terbaik hasil pelatihan
├── .gitignore                         # Mengabaikan file gambar pengujian
└── README.md                          # Dokumentasi proyek
```

---

## 🗃️ Lokasi Cache Dataset

Saat fungsi `load_dataset()` dieksekusi, file dataset dari Hugging Face secara otomatis di-cache di komputer lokal Anda:

- **File mentah / shard unduhan (Hub Cache)**:
  ```text
  C:\Users\<Username>\.cache\huggingface\hub\datasets--<author>--<dataset-name>
  ```
  *Contoh:*
  - `datasets--Parveshiiii--AI-vs-Real`
  - `datasets--Thunderbeast--AI-Generated-vs-Real-Images-Datasets`
  - `datasets--theminji--AI-vs-Real-balanced`

- **File Arrow siap pakai**:
  ```text
  C:\Users\<Username>\.cache\huggingface\datasets
  ```

---

## ⚙️ Persyaratan Sistem & Instalasi

### 1. Prasyarat
- Python 3.11, 3.12, 3.13, atau 3.14 (64-bit)
- Kartu grafis NVIDIA dengan driver terbaru (Disarankan RTX Series)

### 2. Instalasi Dependensi
Jalankan perintah berikut di Terminal / PowerShell untuk memasang PyTorch versi CUDA 12.6 dan pustaka pendukung:

```bash
# Instal PyTorch + TorchVision dengan dukungan CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --upgrade --force-reinstall

# Instal pustaka data science & Hugging Face
pip install datasets pillow numpy matplotlib ipywidgets
```

---

## 🚀 Alur Pelatihan Model

Buka dan jalankan sel di file notebook **`AI vs Real image detection.ipynb`**:

### 1. Inisialisasi Device
Mendeteksi ketersediaan GPU NVIDIA (CUDA):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device yang digunakan: {device} ({torch.cuda.get_device_name(0)})")
```

### 2. Pemuatan Dataset
Dataset dimuat langsung dari Hugging Face Hub:
```python
dataset = load_dataset(
    "theminji/AI-vs-Real-balanced", verification_mode="no_checks"
)
train_hf = dataset["train"]
val_hf = dataset["validation"]
```

### 3. Pipeline Data & Adapter
Mengubah dataset Hugging Face menjadi `torch.utils.data.Dataset` yang kompatibel dengan PyTorch DataLoader:
```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 4. Konfigurasi Arsitektur Model
Mengganti head classifier EfficientNet-B0 menjadi output 2 kelas (`0: AI-Generated`, `1: Real Image`):
```python
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, 2)
)
model = model.to(device)
```

### 5. Pelatihan
Model dilatih menggunakan **CrossEntropyLoss** dan optimizer **AdamW** (`lr=1e-4`, `weight_decay=1e-2`) selama 5 epoch. Model dengan akurasi validasi tertinggi akan disimpan otomatis ke `best_efficientnet_b0_ai_vs_real.pth`.

---

## 🔮 Inferensi & Pengujian Gambar

Untuk menguji gambar kustom baru (misal: `test_image.jpg`), panggil fungsi `predict_image()`:

```python
# Memuat bobot model terbaik
model.load_state_dict(torch.load("best_efficientnet_b0_ai_vs_real.pth"))
model.eval()

label_map = {0: "AI-Generated Image", 1: "Real Image"}


def predict_image(image_path, model):
  img = Image.open(image_path).convert("RGB")
  tensor_img = val_transforms(img).unsqueeze(0).to(device)

  with torch.no_grad():
    output = model(tensor_img)
    probabilities = torch.softmax(output, dim=1)[0]
    confidence, predicted_class = torch.max(probabilities, 0)

  print(f"Hasil Prediksi : {label_map[predicted_class.item()]}")
  print(f"Tingkat Keyakinan: {confidence.item() * 100:.2f}%")


# Contoh pengujian:
predict_image("gani.jpeg", model)
```

---

## 📊 Dataset yang Didukung

| Nama Dataset | Deskripsi | Rekomendasi Penggunaan |
| :--- | :--- | :--- |
| **`theminji/AI-vs-Real-balanced`** | Kumpulan gambar seimbang antara karya seni AI dan manusia (~143k data) | Untuk pelatihan skala besar dengan variasi seni & ilustrasi |

---

## 📄 Lisensi
Proyek ini dibuat untuk keperluan riset dan edukasi dalam bidang Computer Vision dan Deep Learning.
