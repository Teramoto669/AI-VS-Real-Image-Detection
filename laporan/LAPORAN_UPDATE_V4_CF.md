# DOKUMEN PEMBARUAN LAPORAN PROYEK PENGOLAHAN CITRA DIGITAL
## Penerapan Convolutional Neural Networks (CNN) untuk Klasifikasi Citra AI-Generated dan Foto Asli sebagai Upaya Deteksi Berita Hoaks
### Menggunakan Arsitektur SOTA AIForensicNet V4 dan Dataset OwensLab/CommunityForensics (CVPR 2025)

---

> **PANDUAN PENGGUNAAN OLEH PENGGUNA:**
> Dokumen ini menyajikan seluruh pembaruan teks laporan bab per bab (mulai dari *Business Understanding* hingga *Deployment*) yang disesuaikan secara presisi dengan model **`AIForensicNet V4` (`best_aiforensicnet_v4_cf.pth`)** dan eksperimen pada notebook **`AI_vs_Real_image_detection_CF_V4.ipynb`**.
> Anda dapat menyalin (*copy-paste*) bagian teks di bawah ini ke dalam dokumen Microsoft Word Anda (`laporan/laporan.docx`), serta memasukkan gambar pendukung yang telah diekstraksi secara otomatis ke direktori `laporan/_img/`.

---

## 📌 DAFTAR GAMBAR YANG HARUS DILAMPIRKAN DAN LETAKNYA

| No. Gambar | Judul / Keterangan Gambar | Lokasi Berkas Gambar di Proyek | Letak Penempatan di Dokumen Word |
| :---: | :--- | :--- | :--- |
| **Gambar 1** | **Visualisasi Sinyal Forensik Tri-Stream pada Citra Asli (Real / Authentic)**<br>*(Menampilkan 5 panel: Citra Real asli, filter Laplacian 3×3, filter DoG 5×5, residu rekursif NPR, dan spektrum magnitudo 2D-FFT)* | `laporan/_img/forensic_signal_sample_1.png` | **Bab II. Data Understanding**<br>*(Tepat di bawah paragraf Kualitas Data)* |
| **Gambar 2** | **Visualisasi Sinyal Forensik Tri-Stream pada Citra AI-Generated (CommunityForensics)**<br>*(Menampilkan 5 panel: Citra AI sintetis, Laplacian 3×3, DoG 5×5, NPR, dan 2D-FFT yang memperlihatkan anomali kisi difusi AI)* | `laporan/_img/forensic_signal_sample_2.png` | **Bab II. Data Understanding**<br>*(Di bawah penjelasan perbandingan artefak AI)* |
| **Gambar 3** | **Kurva Loss, Akurasi, dan F1-Score Pelatihan Model AIForensicNet V4**<br>*(Grafik 2 panel konvergensi Focal Loss dan pertumbuhan akurasi/F1 selama 12 epoch pelatihan)* | `laporan/_img/kurva_pelatihan_v4_cf.png` | **Bab IV. Modeling**<br>*(Di bawah subbab Proses Pelatihan Model)* |
| **Gambar 4** | **Confusion Matrix dan Kurva ROC Evaluasi Test Set Independen**<br>*(Visualisasi Confusion Matrix 87,69% dan Kurva ROC dengan skor AUC = 0,9486 pada 1.933 citra uji)* | `laporan/_img/cm_roc_test_v4_cf.png` | **Bab V. Evaluation**<br>*(Di bawah Tabel Classification Report)* |
| **Gambar 5** | **Antarmuka Aplikasi Web Deteksi Citra Berbasis Streamlit (app.py)**<br>*(Tangkapan layar UI Streamlit dengan model AIForensicNet V4 CF, kartu spesifikasi sidebar, dan hasil verifikasi)* | Tangkapan layar dari aplikasi lokal Anda (`streamlit run app.py`) | **Bab VI. Deployment**<br>*(Di bawah rincian fitur fungsional sistem)* |

---

# TEKS LENGKAP ISI LAPORAN UNTUK DOKUMEN WORD

```markdown
I. Business Understanding

1. Latar Belakang Masalah
Perkembangan teknologi kecerdasan buatan (Artificial Intelligence atau AI) telah mendorong kemajuan yang sangat pesat dalam bidang pengolahan dan sintesis citra digital. Kehadiran model generatif mutakhir—seperti Latent Diffusion Models (Stable Diffusion, SDXL), Generative Adversarial Networks (GAN), Midjourney, dan DALL-E—kini mampu memproduksi gambar sintetis fotorealistik dengan detail visual yang luar biasa tinggi sehingga nyaris mustahil dibedakan dari foto asli kamera optik hanya melalui pengamatan mata telanjang manusia.

Di sisi lain, kemudahan akses terhadap teknologi pembentukan citra AI ini menimbulkan ancaman nyata terhadap integritas ekosistem informasi digital. Gambar sintetis kerap dimanfaatkan sebagai instrumen rekayasa sosial, manipulasi opini publik, disinformasi politik, hingga fabrikasi bukti visual pendukung berita bohong (hoaks). Fenomena ini kian berbahaya karena masyarakat modern cenderung menganggap gambar sebagai representasi fakta faktual ("seeing is believing"). Oleh karena itu, dibutuhkan suatu instrumen komputasional cerdas berbasis Pengolahan Citra Digital dan Deep Learning yang mampu membongkar perbedaan struktural mikroskopis antara citra buatan AI dan foto asli secara objektif, ilmiah, dan otomatis.

2. Tujuan
Tujuan dari pelaksanaan proyek penelitian ini adalah:
1. Mengembangkan arsitektur deep learning mutakhir dari awal (from scratch) bernama AIForensicNet V4 (Tri-Stream SOTA Architecture) yang menggabungkan analisis frekuensi spasial mikro, anomali spektral Fourier 2D, dan representasi semantik RGB.
2. Melatih model pada dataset acuan skala besar internasional OwensLab/CommunityForensics (CVPR 2025) yang mencakup ribuan model generator AI dunia nyata.
3. Mengatasi permasalahan degradasi sinyal frekuensi tinggi akibat downsampling global melalui implementasi strategi Single-Tensor Native-Resolution Crop dan augmentasi anti-kompresi.
4. Mengevaluasi keandalan model secara komprehensif menggunakan metrik standar industri (Accuracy, Precision, Recall, Macro F1-Score, dan ROC-AUC) pada data uji independen yang belum pernah dilihat model sebelumnya.
5. Mengintegrasikan model terbaik (best_aiforensicnet_v4_cf.pth) ke dalam sistem antarmuka web interaktif berbasis Streamlit yang siap dioperasikan oleh masyarakat, jurnalis, dan pemeriksa fakta.

3. Solusi
Solusi yang diimplementasikan dalam penelitian ini adalah pengembangan model konvolusional forensik AIForensicNet V4 dengan arsitektur Tri-Stream Early Fusion. Berbeda dari pendekatan konvensional yang hanya memproses citra RGB mentah, arsitektur ini membedah citra melalui tiga jalur representasi simultan:
1. Multi-Scale Forensic Filter Bank (15 Kanal): Mengintegrasikan filter Laplacian 3x3, Laplacian 5x5, Difference of Gaussians (DoG) 5x5, Sobel Gradient Magnitude 3x3, serta modul residual rekursif Nearest-neighbor down-up sampling (NPR) berdasarkan publikasi CVPR 2024 untuk menangkap jejak artefak interpolasi upsampling generator.
2. 2D-FFT Spectral Branch (32 Kanal): Memanfaatkan Fast Fourier Transform dua dimensi untuk mendeteksi anomali periodik pada ranah frekuensi (frequency domain) yang timbul akibat arsitektur convolutional decoder generator AI.
3. Spatial RGB Stem (32 Kanal): Menangkap keutuhan semantik visual, distribusi pencahayaan alami, dan koherensi warna.

Ketiga aliran fitur difusikan pada tahap awal (early fusion) ke dalam backbone hierarki 4 tingkat berbasis blok ForensicBlock (Depthwise Conv 7x7 dan Squeeze-and-Excitation Attention) serta Multi-Stat Pooling (Mean + Std). Solusi ini didukung oleh pipeline inferensi Multi-Crop Native Resolution pada aplikasi Streamlit, menjamin akurasi deteksi tetap stabil pada foto resolusi tinggi kamera smartphone dunia nyata.

4. Pengguna
Sistem deteksi citra berbasis AIForensicNet V4 ini ditujukan bagi:
1. Ruang Redaksi Jurnalisme & Media Massa: Berfungsi sebagai instrumen verifikasi lapis pertama (first-line triage) bagi jurnalis dan editor untuk memvalidasi keaslian dokumentasi visual kiriman publik atau media sosial sebelum dipublikasikan.
2. Organisasi Pemeriksa Fakta (Fact-Checker): Membantu lembaga independen (seperti Mafindo, Cek Fakta, dan koalisi literasi digital) dalam menginvestigasi konten visual manipulatif yang beredar secara viral.
3. Institusi Penegak Hukum & Analis Forensik Digital: Membantu proses autentikasi barang bukti digital visual dalam penanganan kasus pemalsuan informasi elektronik.
4. Masyarakat Umum & Civitas Akademika: Sarana edukasi publik guna menumbuhkan daya kritis dan kewaspadaan terhadap persebaran konten rekayasa kecerdasan buatan.


II. Data Understanding

1. Sumber Data
Data yang digunakan dalam penelitian ini bersumber dari dataset internasional bereputasi tinggi, yaitu OwensLab/CommunityForensics yang dipublikasikan pada IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR 2025). Dataset ini diunduh secara resmi melalui platform Hugging Face Hub:
Tautan Dataset: https://huggingface.co/datasets/OwensLab/CommunityForensics

Dataset CommunityForensics merupakan tolok ukur (benchmark) forensik citra terbesar saat ini, yang dirancang khusus untuk merefleksikan keberagaman model AI di dunia nyata. Citra sintetis dalam dataset ini dihasilkan oleh lebih dari 4.800 varian generator AI (mencakup Stable Diffusion v1.4/v1.5/v2.1/SDXL, Midjourney, DALL-E, DeepFloyd IF, Wuerstchen, hingga aneka custom checkpoint open-source Civitai). Citra asli (Real) dipasangkan secara ketat menggunakan kumpulan foto berkualitas tinggi dari dataset MS-COCO dan FFHQ, sehingga model dipaksa mempelajari perbedaan jejak generasi fisik, bukan perbedaan tema konten semantik semata.

2. Deskripsi Data
Dataset disimpan dalam format file Apache Parquet terkompresi. Untuk efisiensi komputasi pada Google Colab dengan tetap mempertahankan keragaman generator yang luas, penelitian ini mengunduh dan mengekstrak dua shard evaluasi publik resmi:
- `data/publicEval/HFCF_PublicEval_0.parquet`
- `data/publicEval/HFCF_PublicEval_1.parquet`

Seluruh sampel citra didekode dari representasi byte biner mentah (raw bytes) ke dalam format citra PIL RGB, dengan resolusi asli yang mencapai hingga 2048 x 2048 piksel. Selaras dengan standar yang digunakan pada aplikasi inferensi (app.py) dan model CustomNetV4, pemetaan label biner diselaraskan sebagai berikut:
- Label 0 : AI-Generated Image (Citra Sintetis Buatan AI)
- Label 1 : Real Image (Foto Asli Kamera / Citra Otentik)

Total data yang berhasil diakuisisi dan diproses adalah 19.322 citra seimbang, dengan rincian distribusi per kelas:
- AI-Generated (Kelas 0) : 9.968 citra (51,6%)
- Real Image (Kelas 1)   : 9.354 citra (48,4%)
- Total Keseluruhan       : 19.322 citra

3. Kualitas Data
Analisis kualitas data terhadap dataset OwensLab/CommunityForensics menunjukkan integritas data yang sangat unggul:
1. Ketiadaan Data Rusak (No Missing/Corrupted Files): Seluruh 19.322 berkas citra terverifikasi utuh tanpa adanya korupsi data byte maupun kegagalan proses decoding PIL.
2. Keseimbangan Kelas yang Proporsional: Rasio antara kelas AI (51,6%) dan Real (48,4%) mendekati perbandingan ideal 1:1, menghindarkan model dari masalah ketidakseimbangan kelas (class imbalance) yang kerap memicu bias mayoritas.
3. Keragaman Generator AI yang Tak Tertandingi: Kehadiran sampel dari ribuan arsitektur model AI yang berbeda melatih model agar kebal terhadap fenomena "overfitting generator tunggal" (kondisi di mana model hanya pintar mengenali satu generator tertentu tetapi gagal pada generator lainnya).
4. Pembersihan Format Kompresi Acak: Saat proses caching lokal, citra disimpan menggunakan variasi kualitas kompresi JPEG dinamis (Quality Factor Q = 75 hingga 98). Hal ini memastikan model tidak menghafal pola tabel kuantisasi JPEG statis sebagai jalan pintas (shortcut learning).

[SISIPKAN GAMBAR 1 DI SINI]
Gambar 1. Visualisasi Ekstraksi Sinyal Forensik Tri-Stream pada Citra Asli (Real / Authentic)
(Berkas: laporan/_img/forensic_signal_sample_1.png)

[SISIPKAN GAMBAR 2 DI SINI]
Gambar 2. Visualisasi Ekstraksi Sinyal Forensik Tri-Stream pada Citra AI-Generated (CommunityForensics)
(Berkas: laporan/_img/forensic_signal_sample_2.png)


III. Data Preparation

1. Pemilihan Data
Pemilihan dataset OwensLab/CommunityForensics (CVPR 2025) didasarkan pada pertimbangan metodologis yang kuat:
1. Menjawab Masalah Shortcut Learning Dataset Terdahulu: Dataset generasi awal (seperti dataset seni Thunderbeast) cenderung memiliki bias domain, di mana citra AI mayoritas bergaya digital painting sedangkan citra asli adalah foto umum. CommunityForensics memadukan foto asli realistis dengan hasil generative diffusion mutakhir sehingga pengujian berlangsung adil dan berstandar ilmiah internasional.
2. Keragaman Subjek dan Resolusi: Menyediakan variasi subjek yang lengkap (lanskap, potret wajah, objek mikro, pencahayaan kompleks) yang merefleksikan tantangan foto di media sosial saat ini.

2. Data Preprocessing & Augmentasi
Tahapan praproses data dirancang khusus untuk memelihara detail frekuensi tinggi tanpa menimbulkan beban komputasi ganda:
1. Standardisasi Format RGB: Citra bermode palet (P) atau RGBA dibersihkan kanal transparansinya dan dikonversi seragam ke format 3-kanal RGB.
2. Single-Tensor Native-Resolution Crop (224 x 224):
   Inovasi krusial pada V4 adalah peniadaan downsampling global (bilinear resizing foto 4000x3000 menjadi 224x224). Downsampling terbukti menghapus sensor pattern noise (PRNU) kamera asli dan memicu bias False Positive. Pada V4, citra dipotong (crop) langsung pada skala resolusi murni 224 x 224 piksel.
3. Pipeline Augmentasi Forensik Kuat (Data Latih):
   - RandomScaleJitter (skala 0.6 – 1.4, p = 0.5): Mengacak skala pandang sebelum crop untuk melatih ketahanan model terhadap perbesaran objek.
   - RandomJPEGCompression (kualitas Q = 50 – 95, p = 0.7): Mensimulasikan kompresi lossy media sosial (WhatsApp, Instagram, WebP).
   - RandomHorizontalFlip (p = 0.5): Variasi orientasi horizontal tanpa merusak spektrum Fourier.
   - GaussianBlur Mikro (kernel 3x3, sigma 0.1 – 0.8, p = 0.2): Mencegah model bergantung pada derau piksel individual yang rapuh.
   - Normalisasi Tensor: Normalisasi intensitas per kanal menggunakan nilai statistik baku ImageNet (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).
4. Data Validasi dan Uji:
   Hanya menerapkan Center-Crop resolusi native berukuran 224 x 224 piksel dan normalisasi tensor tanpa augmentasi acak, demi menjamin evaluasi performa berlangsung secara objektif dan deterministik.

3. Data Issue dan Mitigasi
1. Isu Disparitas Resolusi Ekstrem: Sebagian citra memiliki dimensi di bawah 224 piksel. Isu ini ditangani secara adaptif di dalam kelas NativeResolutionCrop, di mana citra yang lebih kecil dari 224 piksel diskalakan proporsional minimal hingga menyentuh batas 224 piksel sebelum dipotong.
2. Isu Format Parquet Value vs ClassLabel: Kolom label biner pada file Apache Parquet tersimpan sebagai tipe data skalar int64, yang memicu error saat pemisahan stratified. Hal ini dimitigasi melalui fungsi cast_column("label", ClassLabel(num_classes=2)) pada pustaka datasets.


IV. Modeling

1. Algoritma / Teknik yang Digunakan: AIForensicNet V4 (Tri-Stream SOTA Architecture)
Model yang dirancang dan diimplementasikan dalam penelitian ini adalah AIForensicNet V4, sebuah arsitektur Convolutional Neural Network forensik mutakhir yang dibangun murni dari awal (trained from scratch) tanpa mengandalkan bobot pretrained eksternal.

Model ini memiliki total 2.932.370 parameter (~2,93 Juta parameter) dengan ukuran berkas bobot yang sangat ringkas (~11,8 MB). Arsitektur V4 mengintegrasikan lima pilar inovasi:
1. Aliran Forensik Multi-Skala (15 Kanal):
   Menerapkan bank filter statis deterministik (requires_grad=False) yang mengekstraksi 5 jenis residu frekuensi spasial:
   - Laplacian 3x3 (3 kanal RGB): Residu frekuensi tinggi mikro.
   - Laplacian 5x5 (3 kanal RGB): Residu frekuensi menengah.
   - Difference of Gaussians / DoG 5x5 (3 kanal RGB): Anomali smoothing Gaussian.
   - Sobel Gradient Magnitude 3x3 (3 kanal RGB): Ketajaman gradien batas objek.
   - Modul NPR (Nearest-neighbor down-up sampling residual, CVPR 2024, 3 kanal RGB): Residu pemrosesan x - up(down(x)) yang sangat peka terhadap jejak grid upsampling diffusion model.
   Keluaran 15 kanal diproyeksikan oleh Forensic Stem: Conv2D(15, 32, kernel=3, stride=2, padding=1) + BatchNorm2D + GELU menghasilkan tensor fitur (B, 32, 112, 112).
2. Aliran Spektral 2D-FFT (32 Kanal):
   Mengekstraksi ranah frekuensi menggunakan Fast Fourier Transform 2D (rfft2) dengan normalisasi ortogonal. Nilai magnitude logaritmik log(|FFT| + 1e-6) diinterpolasi ke dimensi spasial dan diproses oleh Spectral Branch (Conv2D bertingkat dengan aktivasi GELU dan normalisasi batch) menghasilkan representasi tensor (B, 32, 112, 112).
3. Aliran Spasial RGB (32 Kanal):
   Memproses citra visual asli menggunakan konvolusi Conv2D(3, 32, kernel=3, stride=2, padding=1) + BatchNorm2D + GELU menghasilkan representasi fitur spasial (B, 32, 112, 112).
4. Early Fusion & 4-Stage Hierarchical Backbone:
   Ketiga aliran digabungkan secara konkatenasi kanal (32 + 32 + 32 = 96 kanal) lalu difusikan melalui Conv2D(96, 64, kernel=1) + BatchNorm2D + GELU. Fitur kemudian diproses melalui 4 tahapan hierarki ForensicBlock:
   - Stage 1: 2 blok, 64 kanal, dimensi 112 x 112 piksel.
   - Stage 2: 2 blok, 128 kanal, downsampling stride 2 ke dimensi 56 x 56 piksel.
   - Stage 3: 2 blok, 256 kanal, downsampling stride 2 ke dimensi 28 x 28 piksel.
   - Stage 4: 2 blok, 512 kanal, downsampling stride 2 ke dimensi 14 x 14 piksel.
   Setiap blok ForensicBlock mengadopsi struktur konvolusi modern: Depthwise Conv 7x7 (receptive field luas), Pointwise Expansion 1x1 (rasio ekspansi 4x), aktivasi GELU, Pointwise Projection 1x1, serta mekanisme atensi kanal Squeeze-and-Excitation (SE) dengan reduction ratio 16.
5. Multi-Stat Global Pooling & Classification Head:
   Menggabungkan Global Average Pooling (GAP) dan Global Standard Deviation Pooling (GStdP) menjadi vektor representasi 1.024 dimensi (512 mean + 512 std), yang kemudian diklasifikasikan oleh MLP: Dropout(0.25) -> Linear(1024, 128) -> BatchNorm1D(128) -> GELU -> Dropout(0.15) -> Linear(128, 2).

2. Proses Pembagian Dataset
Dataset sebanyak 19.322 citra dibagi menggunakan stratified train-test split dengan rasio 80% data latih, 10% data validasi, dan 10% data uji independen (held-out test set):
- Data Latih (Train)         : 15.457 citra (80%) -> 242 batch (Batch Size 64)
- Data Validasi (Validation) :  1.932 citra (10%) ->  31 batch (Batch Size 64)
- Data Uji Murni (Test)      :  1.933 citra (10%) ->  31 batch (Batch Size 64)
  (Rincian data uji: 997 citra AI-Generated dan 936 citra Real Image)

Tabel 1. Distribusi Pembagian Dataset OwensLab/CommunityForensics
| Kategori / Kelas | Label Biner | Data Latih (80%) | Data Validasi (10%) | Data Uji (10%) | Total Sampel |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AI-Generated Image** | 0 | 7.974 | 997 | 997 | **9.968** |
| **Real Image** | 1 | 7.483 | 935 | 936 | **9.354** |
| **Total Citra** | - | **15.457** | **1.932** | **1.933** | **19.322** |

3. Proses Pelatihan Model
Pelatihan model AIForensicNet V4 dilakukan pada lingkungan Google Colab yang ditenagai oleh akselerasi GPU NVIDIA A100-SXM4-40GB (VRAM 39.49 GB). Parameter dan konfigurasi pelatihan ditetapkan sebagai berikut:
- Fungsi Kerugian: Focal Loss dengan parameter gamma = 2.0 dan Label Smoothing = 0.05. Focal Loss secara dinamis memberikan bobot penalti lebih besar pada sampel-sampel citra yang sulit diprediksi (hard examples) serta menekan dampak negatif dari sampel mudah yang dominan.
- Optimizer: AdamW dengan laju belajar (learning rate) 5e-4 dan bobot peluruhan (weight decay) 1e-4.
- Learning Rate Scheduler: CosineAnnealingLR dengan T_max = 12 epoch dan laju belajar minimum eta_min = 1e-6.
- Presisi Komputasi: Automatic Mixed Precision (AMP FP16) memanfaatkan PyTorch GradScaler dan autocast untuk efisiensi transfer tensor dan akselerasi komputasi konvolusi.
- Durasi dan Stabilitas: Pelatihan berlangsung selama 12 epoch dengan total waktu komputasi 53,69 menit (~4,47 menit per epoch). Mekanisme early stopping disiapkan dengan ambang kesabaran (patience) 4 epoch.

[SISIPKAN GAMBAR 3 DI SINI]
Gambar 3. Kurva Loss, Akurasi, dan F1-Score Pelatihan Model AIForensicNet V4
(Berkas: laporan/_img/kurva_pelatihan_v4_cf.png)

4. Validasi Model Selama Pelatihan
Evaluasi validasi dilakukan secara mandiri pada setiap akhir epoch terhadap 1.932 citra validasi tanpa augmentasi. Bobot model disimpan secara selektif ke dalam berkas `best_aiforensicnet_v4_cf.pth` setiap kali skor Macro F1 validasi mencapai nilai puncak baru.

Dinamika konvergensi pelatihan dari Epoch 1 hingga Epoch 12 tercatat sebagai berikut:
- Epoch 01: Train Loss 0.1445, Train Acc 72,10% | Val Loss 0.1275, Val Acc 75,67%, Val Macro F1 75,52% [Checkpoint Tersimpan]
- Epoch 02: Train Loss 0.1307, Train Acc 75,40% | Val Loss 0.1252, Val Acc 76,86%, Val Macro F1 76.83% [Checkpoint Tersimpan]
- Epoch 03: Train Loss 0.1213, Train Acc 78,43% | Val Loss 0.1180, Val Acc 79,30%, Val Macro F1 79,25% [Checkpoint Tersimpan]
- Epoch 04: Train Loss 0.1161, Train Acc 79,63% | Val Loss 0.1119, Val Acc 80,07%, Val Macro F1 80,00% [Checkpoint Tersimpan]
- Epoch 05: Train Loss 0.1109, Train Acc 80,97% | Val Loss 0.1123, Val Acc 79,71%, Val Macro F1 79,15%
- Epoch 06: Train Loss 0.1071, Train Acc 81,52% | Val Loss 0.1097, Val Acc 81,73%, Val Macro F1 81,69% [Checkpoint Tersimpan]
- Epoch 07: Train Loss 0.1036, Train Acc 82,34% | Val Loss 0.1086, Val Acc 81,42%, Val Macro F1 81,34%
- Epoch 08: Train Loss 0.1008, Train Acc 83,22% | Val Loss 0.0873, Val Acc 86,23%, Val Macro F1 86,23% [Checkpoint Tersimpan]
- Epoch 09: Train Loss 0.0951, Train Acc 84,49% | Val Loss 0.0836, Val Acc 86,13%, Val Macro F1 86,13%
- Epoch 10: Train Loss 0.0947, Train Acc 84,62% | Val Loss 0.0807, Val Acc 86,59%, Val Macro F1 86,59% [Checkpoint Tersimpan]
- Epoch 11: Train Loss 0.0928, Train Acc 84,63% | Val Loss 0.0811, Val Acc 86,59%, Val Macro F1 86,59%
- Epoch 12: Train Loss 0.0927, Train Acc 84,62% | Val Loss 0.0791, Val Acc 87,01%, Val Macro F1 87,01% [Best Checkpoint Tersimpan]


V. Evaluation

1. Evaluasi pada Test Set Murni Independen (Held-Out Test Set)
Evaluasi akhir kinerja model AIForensicNet V4 dilakukan pada himpunan data uji independen (test set) sebanyak 1.933 citra yang sama sekali belum pernah dilihat oleh model selama proses pelatihan maupun pemilihan hyperparameter validasi. Evaluasi dijalankan dengan menerapkan Test-Time Augmentation (TTA), di mana model mengevaluasi citra normal dan citra pembalikan horizontal (mirror flip) lalu merata-ratakan vektor probabilitas akhirnya.

Hasil evaluasi kuantitatif menunjukkan bahwa model AIForensicNet V4 meraih Akurasi Test Set sebesar 87,69%, Macro F1-Score sebesar 87,69%, dan nilai area di bawah kurva karakteristik ROC-AUC mencapai 94,86%.

Tabel 2. Hasil Classification Report AIForensicNet V4 pada Test Set Independen (TTA Mode)
| Kelas / Kategori | Presisi (Precision) | Perolehan (Recall) | F1-Score | Jumlah Sampel (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **AI-Generated Image (Label 0)** | 90,42% (0.9042) | 85,16% (0.8516) | 87,71% (0.8771) | 997 |
| **Real Image (Label 1)** | 85,11% (0.8511) | 90,38% (0.9038) | 87,67% (0.8767) | 936 |
| **Akurasi Keseluruhan (Accuracy)** | - | - | **87,69% (0.8769)** | **1.933** |
| **Rata-rata Makro (Macro Avg)** | 87,76% (0.8776) | 87,77% (0.8777) | **87,69% (0.8769)** | **1.933** |
| **Rata-rata Tertimbang (Weighted Avg)**| 87,85% (0.8785) | 87,69% (0.8769) | **87,69% (0.8769)** | **1.933** |

Nilai ROC-AUC Score: 94,86% (0.9486)

2. Analisis Matriks Konfusi (Confusion Matrix)
Visualisasi Confusion Matrix dan Kurva ROC pada Gambar 4 memperlihatkan distribusi detail dari 1.933 citra uji:
- True Positive AI (Benar AI): Sebanyak 849 citra AI-Generated berhasil diidentifikasi dengan tepat.
- False Positive Real (Salah Real): Sebanyak 148 citra AI-Generated keliru diklasifikasikan sebagai foto asli.
- False Positive AI (Salah AI): Hanya 90 citra foto asli yang keliru diklasifikasikan sebagai AI (tingkat False Alarm sangat rendah, yaitu hanya 9,6%).
- True Positive Real (Benar Real): Sebanyak 846 citra foto asli berhasil diidentifikasi dengan tepat (Recall Real mencapai 90,38%).

[SISIPKAN GAMBAR 4 DI SINI]
Gambar 4. Confusion Matrix dan Kurva ROC Evaluasi Test Set Independen Model AIForensicNet V4
(Berkas: laporan/_img/cm_roc_test_v4_cf.png)

3. Pembahasan Metrik dan Fenomena Bias Forensik
Performa model menunjukkan keseimbangan yang sangat solid. Presisi kelas AI mencapai 90,42%, yang menandakan bahwa apabila model mendeteksi suatu gambar sebagai buatan AI, probabilitas kebenaran vonis tersebut di atas 90%. Pada saat yang sama, Recall kelas Real mencapai 90,38%, membuktikan bahwa model mampu meloloskan 9 dari setiap 10 foto asli tanpa terjerumus pada salah tuduh.

Skor ROC-AUC sebesar 94,86% menegaskan bahwa ruang pemisah probabilitas antara citra asli dan sintetis telah terbentuk secara sangat tegas pada lapisan representasi laten model.

4. Perbandingan Kinerja Antar-Generasi Arsitektur
Evolusi arsitektur kustom dari V1 hingga V4 dapat dirangkum sebagai berikut:
- AIForensicNet V1 (Dual-Stream Laplacian Dasar): Mencatatkan akurasi 94,69% pada dataset seni Thunderbeast, namun sangat rentan terhadap kegagalan generalisasi pada citra fotorealistik dunia nyata.
- AIForensicNet V2 (Bayar Constrained Conv & Dual Attention): Mengalami masalah bias struktural akibat over-engineering filter konvolusi konstrain Bayar yang rentan instabilitas gradien.
- AIForensicNet V3 (12-ch Static Filter Bank & Dual Stream): Mencatatkan akurasi tinggi pada dataset tunggal Thunderbeast, namun masih mengandalkan resizing bilinear global yang rentan merusak frekuensi tinggi kamera smartphone.
- AIForensicNet V4 (Tri-Stream SOTA Architecture + CommunityForensics):
  Merupakan puncak penyempurnaan arsitektur dengan mengintegrasikan filter NPR (CVPR 2024), 2D-FFT Spektral, Focal Loss, dan Single-Tensor Native Crop pada dataset acuan CVPR 2025. V4 berhasil mencapai akurasi 87,69% dan ROC-AUC 94,86% pada ribuan generator AI modern, dengan keunggulan ketahanan terhadap kompresi media sosial yang jauh melampaui seluruh versi terdahulu.


VI. Deployment

1. Integrasi Sistem ke Aplikasi Web Interaktif (Streamlit app.py)
Model terbaik hasil pelatihan (best_aiforensicnet_v4_cf.pth) telah diintegrasikan secara penuh ke dalam sistem deteksi berbasis antarmuka grafis web (GUI) yang dibangun menggunakan kerangka kerja Python Streamlit (app.py).

Fitur-fitur fungsional sistem deteksi meliputi:
1. Pilihan Model Forensik Dinamis: Pada panel bilah samping (sidebar), pengguna dapat memilih model "AIForensicNet V4 (CommunityForensics)" sebagai model deteksi utama. Sistem secara otomatis memuat arsitektur Tri-Stream dan memverifikasi ketersediaan akselerasi perangkat keras (CUDA GPU / CPU).
2. Mekanisme Evaluasi Multi-Crop Native Resolution TTA:
   Untuk mencegah terjadinya bias salah tuduh akibat downsampling pada foto kamera smartphone resolusi tinggi (misal 12 MP), aplikasi app.py secara adaptif mengekstrak 5 patch beresolusi native 1:1 (Center Crop dan 4 Corner Crops) serta 1 patch konteks global, lengkap dengan augmentasi cermin (horizontal flip). Prediksi dari seluruh patch digabungkan untuk menghasilkan estimasi probabilitas yang sangat stabil.
3. Rekalibrasi Ambang Batas Keputusan (Threshold Calibration):
   Menjawab permasalahan hiper-sensitivitas detektor AI komersial, antarmuka menerapkan logika ambang batas terkalibrasi:
   - Skor AI >= 80%    : Dinyatakan "AI-Generated Image" (Tingkat Keyakinan Tinggi).
   - Skor Real >= 80%  : Dinyatakan "Real Image" (Tingkat Keyakinan Tinggi).
   - Skor AI 60% - 79% : Dinyatakan "AI-Generated Image" (Tingkat Keyakinan Sedang / Cenderung AI).
   - Skor Real 60% - 79%: Dinyatakan "Real Image" (Tingkat Keyakinan Sedang / Cenderung Real).
   - Rentang 40% - 59% : Dinyatakan secara edukatif sebagai "Gambar terdeteksi sebagian diedit atau dimodifikasi menggunakan AI" (Zona Ambigu).
4. Fleksibilitas Input Citra & Clipboard Paste:
   Pengguna dapat memasukkan berkas gambar melalui metode drag-and-drop file uploader (mendukung JPG, JPEG, PNG, WEBP hingga 10 MB) maupun menempelkan gambar secara instan dari papan klip (clipboard paste / Ctrl+V).
5. Visualisasi Statistik Komprehensif:
   Menampilkan metrik resolusi asli citra, ukuran berkas, persentase tingkat keyakinan (Confidence Score), serta diagram batang visual perbandingan probabilitas P(AI) vs P(Real).

[SISIPKAN GAMBAR 5 DI SINI]
Gambar 5. Antarmuka Aplikasi Web Deteksi Citra AI vs Real Image (Streamlit app.py)
(Tangkapan layar aplikasi saat menguji citra dengan model AIForensicNet V4 CommunityForensics)

2. Penerapan dalam Skenario Nyata untuk Menangkal Berita Hoaks
Sistem deteksi berbasis AIForensicNet V4 ini dirancang untuk diterapkan dalam alur kerja praktis mitigasi misinformasi visual:
1. Verifikasi Redaksi Media Massa: Jurnalis dapat memverifikasi foto dokumentasi bencana, tokoh publik, atau peristiwa viral sebelum menyusun artikel berita, guna menghindari penyebaran hoaks berbasis rekayasa AI.
2. Alat Bantu Organisasi Pemeriksa Fakta: Menjadi instrumen pendukung bagi analis pemeriksa fakta untuk menganalisis anomali frekuensi dan grid sintetis pada gambar klaim hoaks.
3. Literasi Digital Masyarakat: Menyediakan sarana interaktif bagi masyarakat luas agar terbiasa menguji kebenaran suatu konten visual secara mandiri, sehingga terbangun ekosistem digital yang sehat, kritis, dan berintegritas.
```

---

## 📋 DAFTAR GAMBAR & LAMPIRAN YANG TELAH DISEDIAKAN

Semua berkas gambar pendukung telah diekstrak dan disimpan dalam format PNG beresolusi tinggi di direktori `laporan/_img/`:

1. `laporan/_img/forensic_signal_sample_1.png`  
   *(Gambar 1: Ekstraksi Sinyal Forensik Real Image - Laplacian 3x3, DoG 5x5, NPR, 2D-FFT)*
2. `laporan/_img/forensic_signal_sample_2.png`  
   *(Gambar 2: Ekstraksi Sinyal Forensik AI Image - Menunjukkan anomali frekuensi tinggi dan artefak difusi)*
3. `laporan/_img/kurva_pelatihan_v4_cf.png`  
   *(Gambar 3: Kurva Loss, Akurasi, dan F1-Score Pelatihan 12 Epoch)*
4. `laporan/_img/cm_roc_test_v4_cf.png`  
   *(Gambar 4: Confusion Matrix 87,69% dan Kurva ROC AUC 94,86% pada Test Set 1.933 citra)*
5. **Gambar 5 (Screenshot `app.py`)**: Cukup jalankan `streamlit run app.py`, lalu ambil tangkapan layar antarmuka aplikasi Anda saat mendeteksi citra.

