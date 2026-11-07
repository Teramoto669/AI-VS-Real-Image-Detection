# Penerapan *Convolutional Neural Networks* (CNN) untuk Klasifikasi Citra *AI-Generated* dan Foto Asli sebagai Upaya Deteksi Berita Hoaks

**Disusun oleh:**

| Nama Mahasiswa | NIM | Peran | Kelas |
| :--- | :---: | :---: | :---: |
| **Rezha Shahidzinda** | 231351139 | Ketua | Malam A |
| **Leonard Putra .S** | 231351073 | Anggota | Malam A |
| **Jafa Shiddiq .B** | 231351068 | Anggota | Malam A |

**PROGRAM STUDI TEKNIK INFORMATIKA**  
**SEKOLAH TINGGI TEKNOLOGI WASTUKANCANA**  
**PURWAKARTA**  
**2026**

---

## I. Business Understanding

### 1. Latar Belakang Masalah
Perkembangan teknologi kecerdasan buatan (*Artificial Intelligence* atau AI) telah mendorong kemajuan yang signifikan dalam bidang pengolahan dan pembuatan citra digital. Salah satu perkembangan tersebut adalah teknologi *AI-Generated Image* berbasis model difusi (*Diffusion Models*) dan *Generative Adversarial Networks* (GAN) generasi terbaru, yang mampu menghasilkan gambar secara otomatis dengan tingkat fotorealisme sangat tinggi sehingga sulit dibedakan dari foto kamera asli hanya melalui pengamatan mata manusia.

Di sisi lain, kemajuan teknologi ini menimbulkan tantangan serius dalam ekosistem informasi digital, khususnya ketika gambar hasil rekayasa AI disalahgunakan untuk menyebarkan berita bohong (hoaks), memanipulasi opini publik, atau memalsukan bukti visual peristiwa. Permasalahan ini semakin kompleks karena model generatif modern (seperti *Stable Diffusion XL, Flux, Midjourney v6,* dan *DALL-E 3*) telah mampu meminimalkan cacat anatomi kasat mata, namun secara matematis tetap meninggalkan jejak anomali mikroskopis pada korelasi piksel bertetangga (*Neighboring Pixel Relationships*), residu derau frekuensi tinggi, serta spektrum frekuensi *Fast Fourier Transform* (2D-FFT). Oleh karena itu, diperlukan pendekatan *Convolutional Neural Networks* (CNN) forensik khusus—yaitu arsitektur **Simplified `AnomalyForensicNet`**—yang dirancang dari nol (*trained from scratch*) untuk mendeteksi jejak forensik citra sintetis secara otomatis dan objektif.

### 2. Tujuan
1. Mengembangkan model klasifikasi citra berbasis *Convolutional Neural Networks* (CNN) arsitektur kustom **Simplified `AnomalyForensicNet`** (`AI_vs_Real_CF_AnomalyForensicNet_Scratch.ipynb`) yang dilatih murni dari nol (*from scratch*) untuk membedakan gambar *AI-Generated* dan foto asli (*Real*).
2. Memanfaatkan dataset benchmark komunitas modern **`OwensLab/CommunityForensics`** sebanyak **20.000 citra seimbang** serta menerapkan teknik *Native Resolution Cropping* dan augmentasi *anti-shortcut learning* agar model mempelajari artefak forensik sejati dan kebal terhadap kompresi media sosial.
3. Mengevaluasi performa model secara independen pada *Test Set* menggunakan metrik *Accuracy*, *Precision*, *Recall*, *Macro F1-Score*, *ROC-AUC*, dan *Confusion Matrix* dengan dukungan *Test-Time Augmentation* (TTA).
4. Mengintegrasikan model terbaik (`best_anomaly_forensicnet_cf.pth`) ke dalam aplikasi web interaktif berbasis *Streamlit* (`app.py`) dengan klasifikasi tiga tingkat keputusan (*AI-Generated Image*, *Sebagian Diedit AI*, dan *Real Image*) sebagai instrumen pendukung verifikasi visual dan pencegahan berita hoaks.

### 3. Solusi
Solusi yang ditawarkan dalam penelitian ini adalah perancangan dan penerapan arsitektur **Simplified `AnomalyForensicNet`** (berkapasitas ringkas **2.383.333 parameter** atau **~2,38 Juta parameter**) yang memadukan tiga aliran informasi (*Tri-Stream Architecture*):
1. **Aliran Forensik Spasial Hibrida (24 Kanal):** Menggabungkan 15 kanal filter deterministik multi-skala (`AnomalyForensicFilterBank`: *Laplacian 3×3, Laplacian 5×5, Difference of Gaussians 5×5, Sobel Gradient Magnitude,* dan *NPR Upsampling Residual*) dengan 9 kanal konvolusi *learnable* `ZeroMeanConv2d` 5×5 untuk menekan konten semantik latar dan mengisolasi residu kamera vs generator AI.
2. **Aliran Spektral Frekuensi 2D-FFT (6 Kanal):** Mengekstrak spektrum *Log-Magnitude* dan *Phase* dari transformasi Fourier 2D (`PhaseMagSpectralStem`) guna menangkap artefak periodik *checkerboard* akibat proses *upsampling* pada dekoder generator AI.
3. **Aliran Konteks Visual RGB (3 Kanal):** Mempertahankan struktur pencahayaan, tekstur permukaan, dan koherensi warna global.

Ketiga aliran tersebut difusikan ke dalam 4 tahap (*4-Stage*) `MixKernelForensicBlock` (kombinasi paralel *Depthwise Convolution* `3×3` dan `7×7`, *Inverted Bottleneck*, `LayerNorm2dClean`, dan `AnomalySqueezeExcitation`) serta diringkas oleh modul **`TopKAnomalyPool2d(topk_ratio=0.15)`** yang menggabungkan *Mean*, *Standard Deviation*, dan *Top-15% aktivasi anomali spasial tertinggi*. Solusi ini diimplementasikan ke dalam aplikasi web *Streamlit* yang mampu mengklasifikasikan citra ke dalam tiga tingkat keputusan: **AI-Generated Image** ($>70\%$ AI), **Real Image** ($>57\%$ Real atau $<43\%$ AI), serta **Sebagian Diedit AI** ($43\% - 70\%$ AI).

### 4. Pengguna
Sistem klasifikasi citra ini ditujukan bagi:
- **Masyarakat Umum:** Sebagai alat periksa mandiri terhadap gambar mencurigakan yang beredar di media sosial atau aplikasi pesan instan sebelum menyebarkannya.
- **Jurnalis, Redaksi Media, dan Pemeriksa Fakta (*Fact-Checker*):** Sebagai instrumen saring lapis pertama (*first-line forensic triage*) untuk memverifikasi keaslian foto berita atau kiriman warga.
- **Tenaga Pendidik, Peneliti, dan Mahasiswa:** Sebagai referensi implementasi arsitektur CNN forensik *from scratch* (`AI_vs_Real_CF_AnomalyForensicNet_Scratch.ipynb`) pada studi pengolahan citra digital dan kecerdasan buatan.

---

## II. Data Understanding

### 1. Sumber Data
Sumber data utama dalam penelitian ini adalah dataset benchmark forensik publik **`OwensLab/CommunityForensics`** yang dikembangkan oleh peneliti dari *University of Michigan* (*OwensLab*) dan tersedia secara terbuka di platform Hugging Face:
- **Repositori Dataset:** [https://huggingface.co/datasets/OwensLab/CommunityForensics](https://huggingface.co/datasets/OwensLab/CommunityForensics)
- **Berkas Shard yang Digunakan:** `data/publicEval/HFCF_PublicEval_0.parquet` dan `data/publicEval/HFCF_PublicEval_1.parquet`.
- **Notebook Eksperimen Utama:** `AI_vs_Real_CF_AnomalyForensicNet_Scratch.ipynb`.

### 2. Deskripsi Data
Dataset diambil dari *shard* evaluasi publik `OwensLab/CommunityForensics` menggunakan pustaka `pyarrow.parquet` dan `huggingface_hub` dengan mekanisme *local caching* (`data/cf_cache/ai` dan `data/cf_cache/real`). Untuk memastikan pelatihan bebas dari bias kelas mayoritas, diekstrak **20.000 citra seimbang sempurna (50:50)** yang terdiri atas **10.000 citra *AI-Generated* (Label `0`)** dan **10.000 citra *Real Image* (Label `1`)**.

Seluruh data kemudian dibagi secara acak berstrata (*stratified random split*) dengan rasio **80% : 10% : 10%** (`SEED = 42`), sebagaimana dirinci pada **Tabel 1**.

**Tabel 1. Distribusi Pembagian Dataset `OwensLab/CommunityForensics` (80:10:10)**

| Kelas / Kategori | Label Proyek | Data Latih (*Train* 80%) | Data Validasi (*Val* 10%) | Data Uji (*Test* 10%) | Total Citra | Persentase |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AI-Generated** | `0` | 8.000 | 1.000 | 1.000 | **10.000** | 50,0% |
| **Real Image** | `1` | 8.000 | 1.000 | 1.000 | **10.000** | 50,0% |
| **Total** | — | **16.000** | **2.000** | **2.000** | **20.000** | **100,0%** |

- **Dimensi & Penyimpanan Citra:** Saat proses ekstraksi dari *shard* Parquet, citra beresolusi sangat besar dibatasi proporsional hingga sisi maksimum `2.048` piksel dan disimpan dalam kualitas JPEG acak dinamis (`quality 80–98`) guna menghilangkan bias format berkas awal.
- **Ukuran Batch DataLoader:** Menggunakan `BATCH_SIZE = 32` (`pin_memory=True`, `num_workers=2`), menghasilkan **500 batch** pada `train_loader`, **63 batch** pada `val_loader`, dan **63 batch** pada `test_loader`.

### 3. Kualitas Data dan Karakteristik Sinyal Forensik
1. **Integritas & Keseimbangan Sempurna:** Seluruh 20.000 citra telah tervalidasi utuh oleh pustaka PIL dan memiliki rasio tepat `1:1` di setiap himpunan (*Train, Val, Test*).
2. **Keragaman Generator Modern:** Mencakup hasil sintesis berbagai generator difusi dan autoregresif modern serta foto kamera asli dengan beragam kondisi pencahayaan dan objek.

![Gambar 1. Sampel Citra Dataset OwensLab/CommunityForensics](_img/afn_cf_cell2_img1.png)  
*Gambar 1. Sampel Citra Dataset `OwensLab/CommunityForensics` (Baris Atas: Real Image | Baris Bawah: AI-Generated).*

![Gambar 2. Visualisasi 7-Panel Ekstraksi Sinyal Forensik Multi-Domain](_img/afn_cf_cell5_img2.png)  
*Gambar 2. Visualisasi 7-Panel Ekstraksi Sinyal Forensik Multi-Domain pada `AnomalyForensicNet` (`Input RGB`, `Laplacian 3×3`, `DoG 5×5`, `Zero-Mean Conv 5×5`, `NPR Residual`, `2D-FFT Log-Magnitude`, dan `2D-FFT Phase Spectrum`).*

---

## III. Data Preparation

### 1. Pemilihan Data
Dataset `OwensLab/CommunityForensics` dipilih karena menyediakan sampel citra AI generasi terbaru yang bebas dari artefak anatomi kasar, sehingga mendorong model untuk benar-benar mempelajari struktur mikroskopis piksel dan spektrum frekuensi alih-alih menghafal gaya warna tertentu.

### 2. Data Preprocessing & Augmentasi Anti-Shortcut Learning
Tahapan prapemrosesan pada `AI_vs_Real_CF_AnomalyForensicNet_Scratch.ipynb` dirancang khusus untuk melindungi sinyal frekuensi tinggi:
1. **Konversi Ruang Warna RGB:** Seluruh citra dikonversi ke mode 3-kanal `RGB` (`ImageFile.LOAD_TRUNCATED_IMAGES = True`).
2. **Pemotongan Resolusi Asli (`NativeResolutionCrop(224)`):**
   - Pengubahan ukuran (*downsampling/resize*) gambar besar secara langsung ke `224 × 224` akan menghancurkan hubungan antar-piksel bertetangga (*Neighboring Pixel Relationships* / NPR) dan merusak spektrum 2D-FFT.
   - Oleh sebab itu, `NativeResolutionCrop` memotong jendela `224 × 224` piksel langsung pada resolusi asli citra (skala piksel `1:1`). Jika sisi terkecil citra $< 224$ piksel, citra diperbesar secara proporsional terlebih dahulu dengan interpolasi *Bilinear*.
   - Pada fase pelatihan (`is_train=True`) digunakan posisi potong acak (*random crop*), sedangkan pada fase validasi dan pengujian (`is_train=False`) digunakan potongan tengah (*center crop*).
3. **Pipeline Augmentasi Anti-Shortcut (`train_transform`):**
   - `RandomScaleJitter(scale_range=(0.6, 1.4), p=0.5)`: Mengacak skala resolusi dengan interpolasi acak (*Bilinear* atau *Bicubic*) sebelum *crop* agar model tidak menghafal jejak skala tetap generator AI.
   - `RandomJPEGCompression(quality_range=(50, 95), p=0.7)`: Melakukan kompresi ulang JPEG secara *on-the-fly* di memori dengan probabilitas 70% agar model tahan terhadap degradasi kompresi media sosial (WhatsApp/Instagram/WebP) dan tidak terjebak pada bias format PNG vs JPEG.
   - `RandomHorizontalFlip(p=0.5)` dan `RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.2)`.
4. **Normalisasi Tensor:** Konversi ke tensor `[0.0, 1.0]` diikuti normalisasi `mean = [0.485, 0.456, 0.406]` dan `std = [0.229, 0.224, 0.225]`.

### 3. Data Issue & Penanganannya
1. **Hilangnya Artefak Mikro Akibat Resize Konvensional:** Diatasi dengan mengganti `transforms.Resize((224, 224))` menjadi `NativeResolutionCrop(224)`.
2. **Bias Kompresi Format Berkas (PNG vs JPEG):** Diatasi melalui *caching* JPEG kualitas acak (`80–98`) serta `RandomJPEGCompression(p=0.7)` saat pelatihan.
3. **Bias Area Sudut Luar (Bokeh/Langit Polos) pada Inferensi:** Pada foto asli beresolusi tinggi, area sudut luar sering kali out-of-focus (*bokeh*) sehingga miskin derau sensor kamera dan memicu alarm palsu pada *Top-K Anomaly Pooling*. Isu ini diselesaikan dengan menyelaraskan inferensi menggunakan **Center Native Resolution Crop (`224 × 224`) + Horizontal Flip TTA**.

---

## IV. Modeling

### 1. Algoritma / Arsitektur: Simplified `AnomalyForensicNet` (From Scratch)
Model utama yang digunakan adalah **Simplified `AnomalyForensicNet`** yang didefinisikan pada `AI_vs_Real_CF_AnomalyForensicNet_Scratch.ipynb` (Cell 4) dan modul produksi `model_custom.py`. Arsitektur ini merupakan jaringan CNN forensik *single-task* yang dilatih murni dari nol (**`2.383.333` parameter *trainable*** / **~2,38 Juta parameter**, ukuran berkas bobot `best_anomaly_forensicnet_cf.pth` sebesar **9,13 MB**).

Penyederhanaan arsitektur dilakukan dengan menghapus cabang dekoder masker inpainting (*multi-task pixel mask head*) dan memusatkan seluruh kapasitas jaringan pada ekstraksi fitur anomalistik tingkat citra (*image-level forensic classification*). Struktur lengkap jaringan terdiri atas 5 blok utama:

1. **Hybrid Multi-Scale Forensic Stem (`24 Kanal -> 32 Kanal`, Resolusi `112 × 112`):**
   - **`AnomalyForensicFilterBank` (15 Kanal Statis Deterministik, `requires_grad=False`, *reflect padding*, `clamp(-5, 5)`):**
     - *Laplacian 3×3* (3 kanal RGB): Mendeteksi derau frekuensi tinggi mikro pada ketetanggaan 1 piksel.
     - *Laplacian 5×5* (3 kanal RGB): Menangkap ketidakwajaran transisi gradien lokal.
     - *Difference of Gaussians / DoG 5×5* (3 kanal RGB): Mengisolasi pita frekuensi menengah bekas penghalusan *denoising* difusi.
     - *Sobel Gradient Magnitude 3×3* ($\sqrt{G_x^2 + G_y^2}$, 3 kanal RGB): Mengekstrak ketajaman mikro tepi objek.
     - *NPR Upsampling Residual* ($(x - \text{BilinearUp}(\text{AvgPool}_{2\times 2}(x))) \times 4.0$, 3 kanal RGB): Mengukur ketergantungan piksel bertetangga akibat lapisan *upsampling* generator.
   - **`ZeroMeanConv2d(3, 9, kernel_size=5)` (9 Kanal *Learnable Zero-Mean*):** Lapisan konvolusi 5×5 yang terus-menerus memproyeksikan bobotnya agar rata-rata per filter bernilai nol ($w \leftarrow w - \mu_w$) pada setiap *forward pass*, sehingga jaringan dapat mempelajari sendiri pola residu derau tanpa terdistraksi oleh warna objek.
   - Ke-24 kanal digabungkan dan diproyeksikan oleh `forensic_stem` (`Conv2d(24, 32, k=3, s=2) -> LayerNorm2dClean(32) -> GELU`).

2. **Phase & Log-Magnitude 2D-FFT Spectral Stem (`PhaseMagSpectralStem`, `6 Kanal -> 32 Kanal`):**
   - Menerapkan transformasi Fourier cepat 2D (`torch.fft.fft2` dan `fftshift`) pada tensor input untuk menghasilkan 3 kanal *Log-Magnitude Spectrum* terstandarisasi dan 3 kanal *Normalized Phase Spectrum* ($\angle F / \pi$).
   - Diproyeksikan melalui `Conv2d(6, 32, k=3, s=2) -> LayerNorm2dClean(32) -> GELU` ke resolusi `112 × 112`.

3. **Aliran Visual RGB & Tri-Stream Stem Fusion (`96 Kanal -> 64 Kanal`):**
   - `rgb_stem` memetakan 3 kanal RGB ke 32 kanal (`112 × 112`).
   - `stem_fusion` menggabungkan ketiga aliran (`32 Forensik + 32 Spektral + 32 RGB = 96 kanal`) melalui `Conv2d(96, 64, k=1) -> LayerNorm2dClean(64) -> GELU` menghasilkan tensor `Stem = (B, 64, 112, 112)`.

4. **4-Stage Hierarchical `MixKernelForensicBlock` Backbone:**
   - Setiap `MixKernelForensicBlock` memiliki dua cabang konvolusi *depthwise* paralel—**`dw3` (`3×3`)** untuk menangkap artefak piksel lokal dan **`dw7` (`7×7`, padding=3)** untuk menangkap ketidakwajaran tekstur jangkauan luas—diikuti normalisasi `LayerNorm2dClean` (mencegah kebocoran statistik antar-sampel dalam *batch*), *Pointwise Inverted Bottleneck* (`expand_ratio = 2`) dengan aktivasi `GELU`, `AnomalySqueezeExcitation(reduction=4)`, serta koneksi *residual skip*.
   - Dimensi tiap tahap:
     - **`Stage 1`**: 2 blok, `64 kanal`, resolusi **`112 × 112`**
     - **`Stage 2`**: 2 blok, `128 kanal` (`stride=2`), resolusi **`56 × 56`**
     - **`Stage 3`**: 2 blok, `256 kanal` (`stride=2`), resolusi **`28 × 28`**
     - **`Stage 4`**: 2 blok, `384 kanal` (`stride=2`), resolusi **`14 × 14`**

5. **`TopKAnomalyPool2d(topk_ratio=0.15)` & Classifier Head (`1.152-D -> 192 -> 2`):**
   - Dari peta fitur `Stage 4` (`384 × 14 × 14 = 196` lokasi spasial), `TopKAnomalyPool2d` menggabungkan **Global Mean (`384-D`)**, **Global Standard Deviation (`384-D`)**, dan **Top-15% Spatial Activation (`384-D`, rata-rata 29 titik spasial terkuat)** menjadi vektor **`1.152` dimensi**.
   - Vektor `1.152-D` diklasifikasikan oleh kepala jaringan: `Dropout(0.30) -> Linear(1152, 192) -> LayerNorm(192) -> GELU -> Dropout(0.15) -> Linear(192, 2)`.

### 2. Proses Pelatihan & Validasi Model
Model dilatih selama **15 epoch** menggunakan GPU **NVIDIA A100-SXM4-80GB** dengan konfigurasi:
- **Loss Function:** Single-Task `FocalLoss(gamma=2.0, label_smoothing=0.05)`.
- **Optimizer:** `AdamW(lr=8e-4, weight_decay=1e-2)` dengan *Gradient Clipping* (`max_norm=1.0`).
- **Scheduler & Presisi:** `CosineAnnealingLR(T_max=15, eta_min=1e-6)` dan *Automatic Mixed Precision* (`torch.amp.autocast("cuda")` + `GradScaler`).

**Tabel 2. Riwayat Lengkap Pelatihan & Validasi Simplified `AnomalyForensicNet` (15 Epoch)**

| Epoch | Durasi (detik) | Train Focal Loss | Train Accuracy (%) | Val Focal Loss | Val Accuracy (%) | Val Macro F1 | Status Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 267,8s | 0.1521 | 69,94% | 0.1240 | 77,50% | 0.7737 | ★ `[BEST SAVED]` |
| **02** | 246,9s | 0.1260 | 77,07% | 0.1109 | 80,10% | 0.8001 | ★ `[BEST SAVED]` |
| **03** | 246,1s | 0.1141 | 80,14% | 0.1019 | 83,45% | 0.8325 | ★ `[BEST SAVED]` |
| **04** | 248,3s | 0.1075 | 82,29% | 0.0970 | 85,25% | 0.8521 | ★ `[BEST SAVED]` |
| **05** | 248,4s | 0.0999 | 83,66% | 0.0809 | 87,30% | 0.8729 | ★ `[BEST SAVED]` |
| **06** | 246,0s | 0.0958 | 84,71% | 0.0725 | 89,40% | 0.8940 | ★ `[BEST SAVED]` |
| **07** | 249,7s | 0.0900 | 85,90% | 0.0783 | 88,10% | 0.8804 | — |
| **08** | 246,2s | 0.0865 | 86,72% | 0.0687 | 89,95% | 0.8994 | ★ `[BEST SAVED]` |
| **09** | 244,3s | 0.0838 | 87,22% | 0.0608 | 91,60% | 0.9160 | ★ `[BEST SAVED]` |
| **10** | 251,1s | 0.0813 | 87,44% | 0.0664 | 90,40% | 0.9038 | — |
| **11** | 248,8s | 0.0786 | 87,80% | 0.0603 | 92,40% | 0.9240 | ★ `[BEST SAVED]` |
| **12** | 244,8s | 0.0773 | 88,23% | 0.0585 | 91,55% | 0.9154 | — |
| **13** | 245,6s | 0.0731 | 89,12% | 0.0569 | 91,75% | 0.9174 | — |
| **14** | 242,1s | 0.0719 | 89,20% | 0.0549 | 92,50% | 0.9250 | ★ `[BEST SAVED]` |
| **15** | 248,3s | **0.0709** | **89,20%** | **0.0535** | **92,55%** | **0.9255** | ★ **`[BEST SAVED]`** |

![Gambar 3. Kurva Perkembangan Focal Loss, Akurasi, dan Macro F1-Score](_img/afn_cf_cell6_img3.png)  
*Gambar 3. Kurva Perkembangan Focal Loss serta Akurasi & Macro F1-Score Selama 15 Epoch Pelatihan.*

---

## V. Evaluation

### 1. Hasil Evaluasi Independen pada *Test Set* (2.000 Citra + Horizontal Flip TTA)
Pengujian akhir dijalankan pada **2.000 citra *Independent Test Set*** (`1.000 AI-Generated` dan `1.000 Real Image`) yang tidak pernah digunakan selama pelatihan maupun pemilihan *checkpoint* validasi. Dengan menerapkan *Horizontal Flip Test-Time Augmentation* (TTA) pada potongan *Center Native Resolution Crop (`224 × 224`)*, diperoleh hasil evaluasi sebagai berikut:

- **Test Accuracy:** **`92,90%` (`0.9290`)**
- **Test Macro F1-Score:** **`92,90%` (`0.9290`)**
- **Test ROC-AUC Score:** **`97,80%` (`0.9780`)**

**Tabel 3. Hasil *Classification Report* Simplified `AnomalyForensicNet` pada *Independent Test Set* (TTA Mode)**

| Kelas / Kategori | Presisi (*Precision*) | *Recall* (Sensitivitas) | *F1-Score* | Jumlah Sampel (*Support*) |
| :--- | :---: | :---: | :---: | :---: |
| **AI-Generated (`0`)** | **92,99%** (`0.9299`) | **92,80%** (`0.9280`) | **92,89%** (`0.9289`) | 1.000 |
| **Real Image (`1`)** | **92,81%** (`0.9281`) | **93,00%** (`0.9300`) | **92,91%** (`0.9291`) | 1.000 |
| **Akurasi Keseluruhan (*Accuracy*)** | — | — | **92,90%** (`0.9290`) | **2.000** |
| **Rata-rata Makro (*Macro Avg*)** | **92,90%** (`0.9290`) | **92,90%** (`0.9290`) | **92,90%** (`0.9290`) | **2.000** |
| **Rata-rata Tertimbang (*Weighted Avg*)** | **92,90%** (`0.9290`) | **92,90%** (`0.9290`) | **92,90%** (`0.9290`) | **2.000** |
| **Skor ROC-AUC** | — | — | **97,80%** (`0.9780`) | **2.000** |

### 2. Analisis *Confusion Matrix* dan Kurva ROC
![Gambar 4. Confusion Matrix dan Kurva ROC pada Independent Test Set](_img/afn_cf_cell7_img4.png)  
*Gambar 4. Confusion Matrix (`Acc = 92,90%`) dan Kurva ROC (`AUC = 0.9780`) pada 2.000 Citra Independent Test Set.*

Berdasarkan *Confusion Matrix* pada **Gambar 4**, rincian prediksi model terhadap 2.000 citra uji independen adalah:
- **True Positive AI (`TP_AI` = 928):** Sebanyak **928 dari 1.000** citra *AI-Generated* berhasil diidentifikasi secara tepat sebagai buatan AI (*Recall AI* = `92,80%`).
- **False Negative AI (`FN_AI` = 72):** Hanya **72** citra *AI-Generated* yang luput dan terklasifikasi sebagai foto asli.
- **True Negative Real (`TN_Real` = 930):** Sebanyak **930 dari 1.000** citra foto kamera asli berhasil dikenali dengan benar sebagai *Real Image* (*Recall Real* = `93,00%`).
- **False Positive AI (`FP_AI` = 70):** Hanya **70** foto asli yang keliru terdeteksi sebagai citra AI.

### 3. Interpretabilitas Peta Aktivasi Hierarkis & Saliensi Anomali Forensik
Untuk memastikan model mengambil keputusan berdasarkan bukti forensik yang valid, dilakukan ekstraksi peta aktivasi dari `Stem` hingga `Stage 4` beserta peta *Anomaly Saliency Overlay* pada sampel data uji:

![Gambar 5. Peta Aktivasi Hierarkis Tiap Stage](_img/afn_cf_cell8_img5.png)  
*Gambar 5. Peta Aktivasi Hierarkis Simplified `AnomalyForensicNet` (`Input RGB` -> `Tri-Stream Stem` -> `Stage 1..4` -> `Anomaly Saliency Overlay`).*

![Gambar 6. Prediksi Sampel Independent Test Set beserta Forensic Saliency Overlay](_img/afn_cf_cell8_img6.png)  
*Gambar 6. Hasil Prediksi Sampel Independent Test Set beserta Peta Lokalisasi Anomali (`Forensic Saliency Overlay`).*

---

## VI. Deployment

### 1. Integrasi Aplikasi Web Interaktif (`app.py` & `model_custom.py`)
Model **Simplified `AnomalyForensicNet`** (`best_anomaly_forensicnet_cf.pth`) telah diintegrasikan ke dalam aplikasi web berbasis **Streamlit** (`app.py`) dengan dukungan arsitektur penuh pada `model_custom.py`:
1. **Sinkronisasi Bobot Presisi Tinggi (`strict=True`):** Seluruh 113 tensor bobot pada `best_anomaly_forensicnet_cf.pth` dimuat secara presisi termasuk `AnomalyForensicFilterBank`, `ZeroMeanConv2d`, `PhaseMagSpectralStem`, `MixKernelForensicBlock`, dan `TopKAnomalyPool2d`.
2. **Inferensi Center Native Resolution Crop (`224 × 224`) + Horizontal Flip TTA:** Menjaga resolusi piksel asli `1:1` pada bagian tengah citra dan merata-ratakan probabilitas *original* serta *horizontal flip*, sehingga bebas dari distorsi *resize* maupun bias area *bokeh* di tepi foto.
3. **Skema Klasifikasi Tiga Tingkat Keputusan (*3-Tier Decision Threshold*):**
   - **AI-Generated Image (`Confidence AI > 70%`):** Citra terdeteksi kuat sebagai hasil sintesis penuh kecerdasan buatan.
   - **Real Image (`Confidence Real > 57%` atau `Confidence AI < 43%`):** Citra terverifikasi memiliki karakteristik derau sensor dan spektrum frekuensi foto kamera asli.
   - **Sebagian Diedit AI (`43% <= Confidence AI <= 70%`):** Kategori peringatan khusus ketika citra memiliki karakteristik campuran—misalnya foto asli yang mengalami penyuntingan generatif sebagian (*AI retouching / inpainting / filter generatif*).

### 2. Penerapan dalam Dunia Nyata
- **Triase Ruang Redaksi & Pemeriksa Fakta:** Membantu jurnalis dan *fact-checker* menyaring bukti visual yang beredar viral dengan cepat sebelum dipublikasikan.
- **Deteksi Manipulasi Hibrida:** Dengan adanya ambang batas zona transisi **Sebagian Diedit AI (`43%–70% AI`)**, pengguna mendapatkan peringatan dini terhadap gambar yang tidak sepenuhnya dibuat dari nol oleh AI, melainkan dimodifikasi sebagian menggunakan fitur *AI generative edit*.
- **Literasi Digital Masyarakat:** Menyediakan antarmuka yang mudah diakses (mendukung unggah berkas dan tempel langsung `Ctrl+V` dari *clipboard*) beserta peta panas anomali forensik sebagai sarana edukasi publik dalam menangkal hoaks visual.
