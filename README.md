# AI vs Real Image Detection 🖼️🤖

Proyek Deep Learning berbasis **PyTorch** dan arsitektur **EfficientNet-B0** untuk mendeteksi dan mengklasifikasikan apakah suatu gambar merupakan buatan kecerdasan buatan (**AI-Generated**) atau foto/gambar asli (**Real**).

---

## 📌 Fitur Utama

- **Transfer Learning State-of-the-Art**: Menggunakan backbone pretrained **EfficientNet-B0** yang ringan, cepat, dan memiliki akurasi tinggi.
- **Presisi Penuh (Full Precision FP32)**: Menggunakan presisi 32-bit floating point tanpa pemotongan kuantisasi demi menjaga integritas sinyal artefak forensik sub-piksel AI yang sangat halus.
- **Regularisasi Head Berlapis (Multi-Dropout)**: Mencegah model mengalami overfitting ke fitur visual tertentu sehingga generalisasi deteksi gambar AI dunia nyata tetap optimal.
- **Integrasi Hugging Face Datasets**: Pemuatan dan pemrosesan dataset langsung dari Hugging Face Hub dengan format streaming/caching otomatis.
- **Penyimpanan Dataset Lokal Proyek**: Opsi `cache_dir="./data"` untuk menyimpan data langsung di direktori proyek.
- **Augmentasi Data Dinamis**: Resizing ke resolusi standar (224x224), *Random Horizontal Flip*, dan *Random Rotation* untuk mencegah overfitting tanpa merusak artefak forensik AI.
- **Evaluasi Komprehensif**: Dilengkapi Classification Report (Precision, Recall, F1-Score) dan visualisasi Confusion Matrix menggunakan `scikit-learn`.
- **Checkpoint Otomatis**: Menyimpan bobot model terbaik (`best_efficientnet_b0_ai_vs_real.pth`) berdasarkan nilai akurasi validasi tertinggi.
- **Fungsi Inferensi Cepat**: Fungsi `predict_image()` bawaan untuk menguji gambar baru dengan output probabilitas keyakinan (*confidence score*).

---

## 🔗 Referensi & Sumber Kode Asli

Kode dasar dan arsitektur alur kerja proyek ini diadaptasi dan dikembangkan dari notebook Kaggle berikut:

- **Original Kaggle Notebook**:  
  👉 [AI vs Real Image Classification (EfficientNet-B0) by Min Khant Tin](https://www.kaggle.com/code/minkhanttin/ai-vs-real-image-classification-efficientnet-b0)

---

## 📊 Dataset yang Digunakan

Proyek ini mendukung dan telah dieksplorasi menggunakan beberapa dataset dari Hugging Face Hub:

| Nama Dataset | Tautan Hugging Face | Deskripsi & Rekomendasi |
| :--- | :--- | :--- |
| **`theminji/AI-vs-Real-balanced`** | [theminji/AI-vs-Real-balanced](https://huggingface.co/datasets/theminji/AI-vs-Real-balanced) | **Dataset Utama Aktif**: Pembagian split `train` (128.762) dan `validation` (14.308) sudah seimbang secara resmi dari pembuat dataset (~143k total gambar). |
| **`Parveshiiii/AI-vs-Real`** | [Parveshiiii/AI-vs-Real](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real) | Sangat direkomendasikan untuk deteksi foto realistis kamera (potret manusia, objek nyata, pemandangan) (~14k gambar). |
| **`Thunderbeast/AI-Generated-vs-Real-Images-Datasets`** | [Thunderbeast/AI-Generated-vs-Real-Images-Datasets](https://huggingface.co/datasets/Thunderbeast/AI-Generated-vs-Real-Images-Datasets) | Dataset ilustrasi dan seni digital (`AiArtData` vs `RealArt`) (~152k gambar). |

---

## 📂 Struktur Proyek

```text
AI VS REAL/
├── AI vs Real image detection.ipynb    # Notebook utama pelatihan & evaluasi model
├── best_efficientnet_b0_ai_vs_real.pth # Bobot model terbaik hasil pelatihan
├── app.py                             # Aplikasi antarmuka Web UI (Streamlit)
├── requirements.txt                   # Daftar pustaka dependensi proyek
├── .streamlit/
│   └── config.toml                    # Konfigurasi Streamlit & batas upload
├── backup/                            # Cadangan file bobot model (.pth)
├── data/                              # Cache dataset lokal proyek (opsional)
├── .gitignore                         # Mengabaikan file gambar uji, backup, & cache dataset
└── README.md                          # Dokumentasi proyek
```

---

## 🗃️ Lokasi Cache Dataset

Dataset yang diunduh dari Hugging Face dapat disimpan di dua tempat:

1. **Di Folder Proyek Lokal (Direkomendasikan)**:
   Dengan menambahkan argumen `cache_dir="./data"` pada `load_dataset()`, file dataset akan tersimpan di dalam folder proyek `data/`.
2. **Di Cache Global Hugging Face (Default)**:
   - **File mentah / shard unduhan (Hub Cache)**:
     ```text
     C:\Users\<Username>\.cache\huggingface\hub\datasets--<author>--<dataset-name>
     ```
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
pip install datasets pillow numpy matplotlib scikit-learn ipywidgets
```

---

## 🚀 Alur Pelatihan Model

Buka dan jalankan sel di file notebook **`AI vs Real image detection.ipynb`**:

### 1. Inisialisasi Device
Mendeteksi ketersediaan GPU NVIDIA (CUDA):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Device yang digunakan: {device} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Device yang digunakan: {device}")
```

### 2. Pemuatan Dataset
Dataset dimuat langsung dari Hugging Face Hub (tersimpan di folder `./data`):
```python
dataset = load_dataset(
    "theminji/AI-vs-Real-balanced",
    cache_dir="./data",
    verification_mode="no_checks",
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

### 4. Konfigurasi Arsitektur Model & Optimizer
Mengonfigurasi classifier head dengan regularisasi dropout untuk 2 kelas (`0: AI-Generated`, `1: Real Image`), serta optimizer AdamW:
```python
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features, 2)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

### 5. Pelatihan Full Precision FP32
Model dilatih menggunakan **CrossEntropyLoss** dan optimizer **AdamW** dalam presisi penuh FP32 selama 5 epoch. Model dengan akurasi validasi tertinggi akan disimpan otomatis ke `best_efficientnet_b0_ai_vs_real.pth`.

### 6. Evaluasi Metrik & Confusion Matrix
Menggunakan model terbaik untuk mengevaluasi dataset validasi secara mendalam dengan **Classification Report** (Precision, Recall, F1-Score) dan visualisasi **Confusion Matrix**.

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
predict_image("test.jpg", model)
```

---

## 🌐 Menjalankan Antarmuka Web Streamlit

Aplikasi UI interaktif berbasis **Streamlit** telah disediakan untuk mempermudah pengujian gambar secara langsung melalui antarmuka web.

### Fitur UI:
1. **Unggah Fleksibel (Drag & Drop & Clipboard Paste)**: 
   - Mendukung drag-and-drop file gambar langsung ke area uploader.
   - Mendukung penempelan citra langsung dari papan klip (*paste clipboard*) melalui tombol khusus maupun pintasan keyboard `Ctrl + V`.
   - Format yang didukung: JPG, JPEG, PNG, dan WEBP.
2. **Pembatasan Ukuran File 2 MB**: Sistem secara otomatis membatasi dan memvalidasi ukuran file (baik dari file maupun clipboard) agar tidak melebihi 2 MB.
3. **Hasil Klasifikasi Real-time**: Menampilkan status **AI-Generated** atau **Real Image** dengan tampilan minimalis dan kontras jelas.
4. **Metrik Keyakinan & Distribusi**: Menampilkan persentase tingkat keyakinan (*confidence score*) serta bilah distribusi probabilitas.
5. **Metadata File**: Menampilkan detail dimensi piksel, format, dan ukuran file gambar masukan.

### Cara Menjalankan:
Jalankan perintah berikut di Terminal / PowerShell pada direktori proyek:

```bash
streamlit run app.py
```

Aplikasi web akan otomatis terbuka di browser pada alamat default: `http://localhost:8501`.

---

## 📄 Lisensi & Atribusi
- Kode referensi asli: [Min Khant Tin on Kaggle](https://www.kaggle.com/code/minkhanttin/ai-vs-real-image-classification-efficientnet-b0) (Apache 2.0 / Open Source).
- Proyek ini dikembangkan lebih lanjut untuk keperluan riset dan edukasi dalam bidang Computer Vision.
