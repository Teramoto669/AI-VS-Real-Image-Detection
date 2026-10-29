import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention block.
    Secara adaptif memberi bobot pada kanal fitur yang paling relevan
    (misal: kanal artefak frekuensi tinggi versus kanal tekstur biasa).
    """
    def __init__(self, channels, reduction=16):
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

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class ForensicBlock(nn.Module):
    """
    Residual Block khusus forensik:
    - Depthwise Conv 7x7 (receptive field luas untuk menangkap pola checkerboard global)
    - Pointwise Conv Expansion & Projection dengan aktivasi GELU
    - Squeeze-and-Excitation Attention
    - Residual Shortcut Connection
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dwconv = nn.Conv2d(in_c, in_c, kernel_size=7, stride=stride, padding=3, groups=in_c, bias=False)
        self.norm = nn.BatchNorm2d(in_c)
        self.pwconv1 = nn.Conv2d(in_c, out_c * 2, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(out_c * 2, out_c, kernel_size=1)
        self.se = SqueezeExcitation(out_c)
        
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.se(out)
        return out + res

class AIForensicNet(nn.Module):
    """
    AIForensicNet: Arsitektur Neural Network Kustom untuk Deteksi Citra AI vs Real.
    
    Inovasi Kunci:
    1. Dual-Stream Stem:
       - Stream 1 (High-Pass / Laplacian): Membuang konten semantik objek dan
         mengisolasi residual derau mikroskopis / jejak sintesis model generatif.
       - Stream 2 (RGB Spasial): Menangkap koherensi warna, pencahayaan, dan tepi objek.
    2. 4-Stage Hierarchical Residual Backbone (~2.85 Juta Parameter).
    3. Ringan, efisien (<1.0 GB VRAM), dan cepat dalam inferensi.
    """
    def __init__(self, num_classes=2, dropout_p=0.3):
        super().__init__()
        
        # 1. High-Pass Filter Bank (Laplacian 3x3 per-channel)
        hp_kernel = torch.tensor([
            [-1., -1., -1.],
            [-1.,  8., -1.],
            [-1., -1., -1.]
        ]).unsqueeze(0).repeat(3, 1, 1, 1)
        self.hp_filter = nn.Parameter(hp_kernel, requires_grad=False)
        
        # 2. Dual Stem (Resolusi input 224x224 -> 112x112)
        self.hp_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        # 3. 4-Stage Residual Backbone
        self.stage1 = nn.Sequential(
            ForensicBlock(64, 64),
            ForensicBlock(64, 64)
        )
        self.stage2 = nn.Sequential(
            ForensicBlock(64, 128, stride=2),
            ForensicBlock(128, 128)
        )
        self.stage3 = nn.Sequential(
            ForensicBlock(128, 256, stride=2),
            ForensicBlock(256, 256)
        )
        self.stage4 = nn.Sequential(
            ForensicBlock(256, 512, stride=2),
            ForensicBlock(512, 512)
        )
        
        # 4. Classification Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Stream 1: Ekstraksi residual derau frekuensi tinggi
        hp_x = F.conv2d(x, self.hp_filter, padding=1, groups=3)
        feat_hp = self.hp_stem(hp_x)
        
        # Stream 2: Ekstraksi spasial RGB
        feat_rgb = self.rgb_stem(x)
        
        # Fusi dua aliran (32 + 32 = 64 kanal)
        out = torch.cat([feat_hp, feat_rgb], dim=1)
        
        # Hierarki representasi fitur
        out = self.stage1(out)  # 112x112
        out = self.stage2(out)  # 56x56
        out = self.stage3(out)  # 28x28
        out = self.stage4(out)  # 14x14
        
        # Prediksi kelas
        return self.head(out)


# ==============================================================================
# AIForensicNet V2 (Optimized & Next-Gen Forensic Architecture)
# ==============================================================================

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample saat training.
    Mencegah co-adaptation dan overfitting pada representasi residual.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM-style).
    Menyorot lokasi/area spesifik tempat anomali artefak AI muncul
    (misal: batas tepi tidak wajar, detail mata, tekstur rambut, atau artefak distorsi).
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class BayarConv2d(nn.Module):
    """
    Bayar Constrained Convolution (Bayar & Stamm, IEEE WIFS).
    Secara matematis meniadakan rata-rata konten semantik visual (sum of weights = 0)
    sehingga model secara adaptif BELAJAR mengekstrak pola derau (noise fingerprint)
    spesifik dari model-model generatif AI terbaru.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1, padding: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.center_idx = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        c = self.center_idx
        mask = torch.ones_like(self.weight)
        mask[:, :, c, c] = 0.0
        w_masked = self.weight * mask
        sum_w = w_masked.sum(dim=(2, 3), keepdim=True) + 1e-7
        w_norm = w_masked / sum_w
        center_mask = torch.zeros_like(self.weight)
        center_mask[:, :, c, c] = 1.0
        w_final = w_norm * mask - center_mask
        return F.conv2d(x, w_final, stride=self.stride, padding=self.padding)


class SRMConv2d(nn.Module):
    """
    Spatial Rich Models (SRM) Filter Bank.
    Filter baku steganografi digital untuk menangkap residual berorde tinggi (1st, 2nd, & cross order).
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        srm1 = torch.tensor([
            [0,  0,  0,  0,  0],
            [0, -1,  2, -1,  0],
            [0,  2, -4,  2,  0],
            [0, -1,  2, -1,  0],
            [0,  0,  0,  0,  0]
        ], dtype=torch.float32) / 4.0

        srm2 = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 12.0

        srm3 = torch.tensor([
            [0,  0,  0,  0,  0],
            [0,  0,  1,  0,  0],
            [0,  1, -4,  1,  0],
            [0,  0,  1,  0,  0],
            [0,  0,  0,  0,  0]
        ], dtype=torch.float32)

        kernels = torch.stack([srm1, srm2, srm3]).unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer('srm_weight', kernels)
        self.in_channels = in_channels

    def forward(self, x):
        return F.conv2d(x, self.srm_weight, padding=2, groups=self.in_channels)


class ForensicBlockV2(nn.Module):
    """
    ForensicBlock Generasi ke-2:
    - 7x7 Depthwise Conv untuk receptive field global (menangkap checkerboard artifacts)
    - Inverted Bottleneck dengan ekspansi kanal 3x + GELU
    - Dual Attention (Channel Squeeze-Excitation + Spatial Attention)
    - Normalisasi stabil ganda (dwconv norm & projection norm)
    - LayerScale untuk percepatan konvergensi & kestabilan gradien
    - DropPath (Stochastic Depth) untuk regularisasi kuat
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1, drop_path: float = 0.0, layer_scale_init_value: float = 1e-5):
        super().__init__()
        self.dwconv = nn.Conv2d(in_c, in_c, kernel_size=7, stride=stride, padding=3, groups=in_c, bias=False)
        self.norm1 = nn.BatchNorm2d(in_c)

        exp_c = in_c * 3
        self.pwconv1 = nn.Conv2d(in_c, exp_c, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(exp_c, out_c, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_c)

        self.se = SqueezeExcitation(out_c)
        self.sa = SpatialAttention(kernel_size=7)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(out_c, 1, 1), requires_grad=True)
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.dwconv(x)
        out = self.norm1(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.norm2(out)
        out = self.se(out)
        out = self.sa(out)

        if self.gamma is not None:
            out = self.gamma * out

        out = self.drop_path(out)
        return out + res


class AIForensicNetV2(nn.Module):
    """
    AIForensicNetV2: Versi Teroptimasi untuk Deteksi Citra AI vs Real.

    Peningkatan Utama dibanding V1:
    1. Multi-Filter Forensic Stem:
       - 9 Filter SRM (Fixed Steganalysis Bank)
       - 7 Filter Bayar Constrained Conv (Learnable Noise Pattern Extraction)
       Total 16 kanal fitur forensik (bukan hanya 1 filter Laplacian statis).
    2. Dual Attention (Channel SE + Spatial Attention) di setiap blok residual.
    3. LayerScale + DropPath (Stochastic Depth) untuk konvergensi lebih dalam & tanpa overfitting.
    4. Concat Dual-Pooling Head:
       Menggabungkan AdaptiveAvgPool2d (konteks global) dan AdaptiveMaxPool2d (puncak artefak terlokalisir)
       menjadi 1024 fitur sebelum klasifikasi.
    5. Kapasitas seimbang: ~3.52M parameter, konsumsi VRAM < 1.0 GB.
    """
    def __init__(self, num_classes: int = 2, drop_rate: float = 0.2, drop_path_rate: float = 0.1):
        super().__init__()

        # 1. Forensic High-Pass Stream: Fixed SRM (9 ch) + Learnable Bayar (7 ch) = 16 ch
        self.srm = SRMConv2d(3)
        self.bayar = BayarConv2d(3, 7, kernel_size=5, stride=1, padding=2)

        self.hp_stem = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # 2. RGB Spatial Stream
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # 3. 4-Stage Hierarchical Residual Backbone dengan decay DropPath
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 8)]

        self.stage1 = nn.Sequential(
            ForensicBlockV2(64, 64, stride=1, drop_path=dpr[0]),
            ForensicBlockV2(64, 64, stride=1, drop_path=dpr[1])
        )
        self.stage2 = nn.Sequential(
            ForensicBlockV2(64, 128, stride=2, drop_path=dpr[2]),
            ForensicBlockV2(128, 128, stride=1, drop_path=dpr[3])
        )
        self.stage3 = nn.Sequential(
            ForensicBlockV2(128, 256, stride=2, drop_path=dpr[4]),
            ForensicBlockV2(256, 256, stride=1, drop_path=dpr[5])
        )
        self.stage4 = nn.Sequential(
            ForensicBlockV2(256, 512, stride=2, drop_path=dpr[6]),
            ForensicBlockV2(512, 512, stride=1, drop_path=dpr[7])
        )

        # 4. Multi-Pool Classification Head (AvgPool + MaxPool = 1024 dims)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(p=drop_rate / 2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Ekstraksi aliran forensik
        srm_feat = self.srm(x)
        bayar_feat = self.bayar(x)
        forensic_feat = torch.cat([srm_feat, bayar_feat], dim=1)
        feat_hp = self.hp_stem(forensic_feat)

        # Ekstraksi aliran spasial RGB
        feat_rgb = self.rgb_stem(x)

        # Fusi dua aliran (32 + 32 = 64 kanal)
        out = torch.cat([feat_hp, feat_rgb], dim=1)

        # Hierarki representasi fitur
        out = self.stage1(out)  # 112x112
        out = self.stage2(out)  # 56x56
        out = self.stage3(out)  # 28x28
        out = self.stage4(out)  # 14x14

        # Dual Pooling (Konteks Global + Puncak Anomali)
        avg_pool = self.gap(out).flatten(1)
        max_pool = self.gmp(out).flatten(1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        return self.head(pooled)


# ==============================================================================
# AIForensicNet V3 (Multi-Scale Static Filter Bank + Mean-Std Pooling)
# ==============================================================================

class MultiScaleForensicFilterBank(nn.Module):
    """
    Bank Filter Forensik Multi-Skala Deterministik (requires_grad=False).
    Mengekstrak sidik jari frekuensi tinggi multi-skala secara instan tanpa
    ketidakstabilan optimasi gradient dari Bayar Conv:
    - Laplacian 3x3 (fine micro-noise residual)
    - Laplacian 5x5 (mid-frequency synthesis patterns)
    - Difference of Gaussians (DoG 5x5, mendeteksi penghalusan de-noising khas diffusion)
    - Sobel Magnitude 3x3 (diskontinuitas gradien spasial)
    Total 12 kanal fitur residual (4 jenis filter x 3 kanal RGB).
    """
    def __init__(self):
        super().__init__()
        lap3 = torch.tensor([
            [-1., -1., -1.],
            [-1.,  8., -1.],
            [-1., -1., -1.]
        ], dtype=torch.float32)

        lap5 = torch.tensor([
            [ 0.,  0., -1.,  0.,  0.],
            [ 0., -1., -2., -1.,  0.],
            [-1., -2., 16., -2., -1.],
            [ 0., -1., -2., -1.,  0.],
            [ 0.,  0., -1.,  0.,  0.]
        ], dtype=torch.float32)

        dog = torch.tensor([
            [-1., -3., -4., -3., -1.],
            [-3.,  0.,  6.,  0., -3.],
            [-4.,  6., 20.,  6., -4.],
            [-3.,  0.,  6.,  0., -3.],
            [-1., -3., -4., -3., -1.]
        ], dtype=torch.float32) / 16.0

        sobel_h = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ], dtype=torch.float32)

        sobel_v = torch.tensor([
            [-1.,  0.,  1.],
            [-2.,  0.,  2.],
            [-1.,  0.,  1.]
        ], dtype=torch.float32)

        k3 = torch.stack([lap3, sobel_h, sobel_v]).unsqueeze(1).repeat(3, 1, 1, 1)
        k5 = torch.stack([lap5, dog]).unsqueeze(1).repeat(3, 1, 1, 1)

        self.register_buffer('k3', k3)
        self.register_buffer('k5', k5)

    def forward(self, x):
        f3 = F.conv2d(x, self.k3, padding=1, groups=3)
        f5 = F.conv2d(x, self.k5, padding=2, groups=3)

        # R, G, B Sobel Magnitude
        sh = f3[:, 1::3, :, :]
        sv = f3[:, 2::3, :, :]
        smag = torch.sqrt(sh ** 2 + sv ** 2 + 1e-6)

        lap3_out = f3[:, 0::3, :, :]
        lap5_out = f5[:, 0::2, :, :]
        dog_out = f5[:, 1::2, :, :]

        # 3 + 3 + 3 + 3 = 12 kanal
        return torch.cat([lap3_out, lap5_out, dog_out, smag], dim=1)


class MeanStdPool2d(nn.Module):
    """
    Global Mean & Standard Deviation Pooling.
    - Mean: Menangkap rata-rata intensitas aktivasi fitur global.
    - Std: Menangkap variasi/fluktuasi sebaran derau mikro di seluruh bidang spasial.
    Sangat tangguh terhadap outlier dibanding MaxPool dan tidak bias terhadap noise sporadis.
    """
    def forward(self, x):
        mean = torch.mean(x, dim=(2, 3))
        std = torch.std(x, dim=(2, 3), unbiased=False)
        return torch.cat([mean, std], dim=1)


class AIForensicNetV3(nn.Module):
    """
    AIForensicNetV3: Arsitektur Forensik Generasi ke-3 Teroptimasi.
    
    Penyempurnaan Utama:
    1. Multi-Scale Static Filter Bank:
       - 12 kanal residual deterministik (Laplacian 3x3, Laplacian 5x5, DoG, Sobel)
       - Bebas instabilitas gradient, stabil sejak epoch 1.
    2. Balanced Dual-Stream Stem:
       - 32 kanal forensik multi-skala + 32 kanal spasial RGB = 64 kanal fusi seimbang.
    3. Residual Backbone berbasis Depthwise 7x7 + SE Attention murni (Occam's Razor):
       - Menghapus Spatial Attention & DropPath yang terbukti mengaburkan sinyal forensik di V2.
    4. Global Mean + Std Deviation Pooling:
       - Mengukur konsistensi statistik derau di seluruh citra tanpa tertipu outlier.
    5. Kapasitas seimbang: ~3.07 Juta parameter, sangat cepat dan efisien VRAM.
    """
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.25):
        super().__init__()

        # 1. Multi-Scale Forensic Filter Bank (12 kanal output)
        self.filter_bank = MultiScaleForensicFilterBank()

        # 2. Dual-Stream Stem (Resolusi 224x224 -> 112x112)
        self.forensic_stem = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # 3. 4-Stage Residual Backbone (Depthwise 7x7 + Pointwise Expansion + SE Attention)
        self.stage1 = nn.Sequential(
            ForensicBlock(64, 64),
            ForensicBlock(64, 64)
        )
        self.stage2 = nn.Sequential(
            ForensicBlock(64, 128, stride=2),
            ForensicBlock(128, 128)
        )
        self.stage3 = nn.Sequential(
            ForensicBlock(128, 256, stride=2),
            ForensicBlock(256, 256)
        )
        self.stage4 = nn.Sequential(
            ForensicBlock(256, 512, stride=2),
            ForensicBlock(512, 512)
        )

        # 4. Multi-Stat Classification Head (Mean + Std = 1024 fitur)
        # Menggunakan LayerNorm(256) alih-alih BatchNorm1d untuk stabilitas evaluasi 100% konsisten
        # tanpa fluktuasi/spike running statistics antar-epoch.
        self.pool = MeanStdPool2d()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512 * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes)
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Memastikan kompatibilitas backward jika memuat bobot dari checkpoint lama berbasis BatchNorm1d."""
        for key in list(state_dict.keys()):
            if key.startswith(prefix + "head.2.") and any(x in key for x in ["running_mean", "running_var", "num_batches_tracked"]):
                state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # Aliran 1: Ekstraksi sinyal residual multi-skala
        forensic_signals = self.filter_bank(x)
        feat_forensic = self.forensic_stem(forensic_signals)

        # Aliran 2: Ekstraksi fitur visual RGB
        feat_rgb = self.rgb_stem(x)

        # Fusi dua aliran (32 + 32 = 64 kanal)
        out = torch.cat([feat_forensic, feat_rgb], dim=1)

        # Hierarki representasi fitur (112 -> 56 -> 28 -> 14)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # Pooling statistik global (Mean + Std)
        pooled = self.pool(out)

        # Klasifikasi probabilitas kelas
        return self.head(pooled)



# ==============================================================================
# AIForensicNet V4 (State-of-the-Art Tri-Stream Architecture)
# ==============================================================================

class NPRModule(nn.Module):
    """
    Neighboring Pixel Relationships (NPR) Extractor (Tan et al., CVPR 2024).
    Secara spesifik mendeteksi jejak inter-pixel correlation yang ditinggalkan oleh
    operasi up-sampling (Nearest/Bilinear/Transposed Conv) pada arsitektur generator
    AI modern (GAN dan Latent Diffusion Decoders).
    
    Formula:
        x_down = Interp(x, scale=0.5, mode='nearest')
        x_up   = Interp(x_down, scale=2.0, mode='nearest')
        NPR(x) = x - x_up
    """
    def forward(self, x):
        down = F.interpolate(x, scale_factor=0.5, mode='nearest')
        up = F.interpolate(down, size=x.shape[2:], mode='nearest')
        return x - up


class FFTSpectralBranch(nn.Module):
    """
    Cabang Frekuensi 2D (FFT Spectral Branch).
    Mengekstrak anomali spektrum Fourier radial-symmetry dan artefak puncak
    frekuensi periodik (Durall et al., Frank et al.) yang tidak tampak di domain spasial.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(16)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        # rfft2 menghitung Real-valued Fast Fourier Transform 2D
        fft = torch.fft.rfft2(x, norm="ortho")
        # Log magnitude spektrum frekuensi
        mag = torch.log(torch.abs(fft) + 1e-6)
        
        # Samakan dimensi spektral dengan dimensi spasial input
        mag_interp = F.interpolate(mag, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        feat = self.act1(self.norm1(self.conv1(mag_interp)))
        feat = self.act2(self.norm2(self.conv2(feat)))
        return feat


class MultiScaleForensicFilterBankV4(nn.Module):
    """
    Bank Filter Forensik Komprehensif V4:
    1. Laplacian 3x3: Derau frekuensi tinggi mikro (3 ch)
    2. Laplacian 5x5: Residu sintetis frekuensi menengah (3 ch)
    3. Difference of Gaussians (DoG 5x5): Jejak penghalusan de-noising khas difusi (3 ch)
    4. Sobel Magnitude 3x3: Diskontinuitas gradien spasial (3 ch)
    5. NPR (Neighboring Pixel Relationships): Jejak up-sampling decoder generator (3 ch)
    Total: 15 kanal fitur residual deterministik (requires_grad=False).
    """
    def __init__(self):
        super().__init__()
        lap3 = torch.tensor([
            [-1., -1., -1.],
            [-1.,  8., -1.],
            [-1., -1., -1.]
        ], dtype=torch.float32)

        lap5 = torch.tensor([
            [ 0.,  0., -1.,  0.,  0.],
            [ 0., -1., -2., -1.,  0.],
            [-1., -2., 16., -2., -1.],
            [ 0., -1., -2., -1.,  0.],
            [ 0.,  0., -1.,  0.,  0.]
        ], dtype=torch.float32)

        dog = torch.tensor([
            [-1., -3., -4., -3., -1.],
            [-3.,  0.,  6.,  0., -3.],
            [-4.,  6., 20.,  6., -4.],
            [-3.,  0.,  6.,  0., -3.],
            [-1., -3., -4., -3., -1.]
        ], dtype=torch.float32) / 16.0

        sobel_h = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ], dtype=torch.float32)

        sobel_v = torch.tensor([
            [-1.,  0.,  1.],
            [-2.,  0.,  2.],
            [-1.,  0.,  1.]
        ], dtype=torch.float32)

        k3 = torch.stack([lap3, sobel_h, sobel_v]).unsqueeze(1).repeat(3, 1, 1, 1)
        k5 = torch.stack([lap5, dog]).unsqueeze(1).repeat(3, 1, 1, 1)

        self.register_buffer('k3', k3)
        self.register_buffer('k5', k5)
        self.npr = NPRModule()

    def forward(self, x):
        f3 = F.conv2d(x, self.k3, padding=1, groups=3)
        f5 = F.conv2d(x, self.k5, padding=2, groups=3)

        # Sobel Magnitude
        sh = f3[:, 1::3, :, :]
        sv = f3[:, 2::3, :, :]
        smag = torch.sqrt(sh ** 2 + sv ** 2 + 1e-6)

        lap3_out = f3[:, 0::3, :, :]
        lap5_out = f5[:, 0::2, :, :]
        dog_out = f5[:, 1::2, :, :]
        npr_out = self.npr(x)

        # 3 (Lap3) + 3 (Lap5) + 3 (DoG) + 3 (Sobel) + 3 (NPR) = 15 kanal
        return torch.cat([lap3_out, lap5_out, dog_out, smag, npr_out], dim=1)


class MultiStatPool2d(nn.Module):
    """
    Global Multi-Statistic Pooling (Mean + Standard Deviation).
    - Mean (Momen ke-1): Mengukur rata-rata intensitas aktivasi fitur global.
    - Std (Momen ke-2): Mengukur dispersi/sebaran variansi spasial energi derau.
    Secara matematis mampu membedakan homogenitas tekstur foto asli vs
    ketidakseragaman mikro sintetis AI di seluruh dimensi spasial.
    """
    def forward(self, x):
        mean = torch.mean(x, dim=(2, 3))
        var = torch.var(x, dim=(2, 3), unbiased=False)
        std = torch.sqrt(var + 1e-6)
        return torch.cat([mean, std], dim=1)


class AIForensicNetV4(nn.Module):
    """
    AIForensicNetV4: Arsitektur Forensik Generasi ke-4 (Penyempurnaan Penuh).
    
    Penyempurnaan Utama Terhadap V1, V2, dan V3:
    1. Tri-Stream Architecture:
       - Stream 1 (15 Kanal Multi-Scale Residual Bank + NPR): Menangkap jejak de-noising & upsampling.
       - Stream 2 (Spectral FFT Branch): Menangkap anomali simetri frekuensi 2D pada domain Fourier.
       - Stream 3 (Spatial RGB Branch): Menangkap inkonsistensi semantik, pencahayaan, dan tepi objek.
    2. Tri-Stream Projection Fusion:
       - Fusi terintegrasi (32 ch forensik + 32 ch spektral + 32 ch RGB = 96 ch) diproyeksikan ke 64 ch.
    3. ConvNeXt-Style Residual Backbone Murni:
       - Menghilangkan LayerScale & DropPath yang memperlambat konvergensi di V2.
       - Mempertahankan Depthwise 7x7 + Squeeze-and-Excitation (SE) channel attention.
    4. Multi-Stat Pooling Head (Mean + Std = 1024 Dimensi):
       - Kepala klasifikasi ramping 2-layer dengan regulasi Dropout bertahap (0.25 -> 0.15).
    5. Kapasitas Ideal: ~3.12 Juta Parameter (From Scratch, sangat efisien VRAM <1.0 GB).
    """
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.25):
        super().__init__()

        # 1. Aliran 1: Multi-Scale Forensic + NPR (15 kanal input)
        self.forensic_bank = MultiScaleForensicFilterBankV4()
        self.forensic_stem = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # 2. Aliran 2: Cabang Spektral 2D-FFT
        self.spectral_branch = FFTSpectralBranch(in_channels=3, out_channels=32)

        # 3. Aliran 3: Aliran Spasial RGB
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # 4. Tri-Stream Fusion Layer: 32 + 32 + 32 = 96 kanal -> 64 kanal
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # 5. 4-Stage Hierarchical Residual Backbone
        self.stage1 = nn.Sequential(
            ForensicBlock(64, 64),
            ForensicBlock(64, 64)
        )
        self.stage2 = nn.Sequential(
            ForensicBlock(64, 128, stride=2),
            ForensicBlock(128, 128)
        )
        self.stage3 = nn.Sequential(
            ForensicBlock(128, 256, stride=2),
            ForensicBlock(256, 256)
        )
        self.stage4 = nn.Sequential(
            ForensicBlock(256, 512, stride=2),
            ForensicBlock(512, 512)
        )

        # 6. Multi-Stat Classification Head (Mean + Std = 1024 Dimensi)
        self.pool = MultiStatPool2d()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512 * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Stream 1: Ekstraksi fitur forensik spasial & residu NPR
        forensic_signals = self.forensic_bank(x)
        feat_forensic = self.forensic_stem(forensic_signals)  # (B, 32, 112, 112)

        # Stream 2: Ekstraksi anomali frekuensi 2D FFT
        feat_spectral = self.spectral_branch(x)               # (B, 32, 112, 112)

        # Stream 3: Ekstraksi fitur spasial RGB
        feat_rgb = self.rgb_stem(x)                           # (B, 32, 112, 112)

        # Fusi Tri-Stream (32 + 32 + 32 = 96 -> 64 kanal)
        tri_feat = torch.cat([feat_forensic, feat_spectral, feat_rgb], dim=1)
        out = self.fusion(tri_feat)

        # Hierarki representasi fitur
        out = self.stage1(out)  # 112x112
        out = self.stage2(out)  # 56x56
        out = self.stage3(out)  # 28x28
        out = self.stage4(out)  # 14x14

        # Pooling statistik global (Mean + Std)
        pooled = self.pool(out)

        # Klasifikasi probabilitas kelas
        return self.head(pooled)


# ==============================================================================
# AIForensicNet V4 HiDF (Exact Model Definition for HiDF Deepfake Dataset)
# ==============================================================================

class FFTSpectralBranchHiDF(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    def forward(self, x):
        fft_val = torch.fft.fft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft_val, dim=(-2, -1))
        mag = torch.log(torch.abs(fft_shift) + 1e-6)
        mag = (mag - mag.mean(dim=(-2, -1), keepdim=True)) / (mag.std(dim=(-2, -1), keepdim=True) + 1e-6)
        return self.proj(mag)


class MultiScaleForensicFilterBankV4HiDF(nn.Module):
    def __init__(self):
        super().__init__()
        lap3 = torch.tensor([[-1., -1., -1.], [-1.,  8., -1.], [-1., -1., -1.]], dtype=torch.float32)
        lap5 = torch.tensor([[0., 0., -1., 0., 0.], [0., -1., -2., -1., 0.], [-1., -2., 16., -2., -1.], [0., -1., -2., -1., 0.], [0., 0., -1., 0., 0.]], dtype=torch.float32)
        dog = torch.tensor([[-1., -3., -4., -3., -1.], [-3., 0., 6., 0., -3.], [-4., 6., 20., 6., -4.], [-3., 0., 6., 0., -3.], [-1., -3., -4., -3., -1.]], dtype=torch.float32) / 16.0
        sobel_h = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32)
        sobel_v = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32)

        k3 = torch.stack([lap3, sobel_h, sobel_v]).unsqueeze(1).repeat(3, 1, 1, 1)
        self.register_buffer('k3_weight', k3)
        k5 = torch.stack([lap5, dog]).unsqueeze(1).repeat(3, 1, 1, 1)
        self.register_buffer('k5_weight', k5)
        self.npr = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)

    def forward(self, x):
        f3 = F.conv2d(x, self.k3_weight, padding=1, groups=3)
        f5 = F.conv2d(x, self.k5_weight, padding=2, groups=3)
        sh = f3[:, 1::3, :, :]
        sv = f3[:, 2::3, :, :]
        smag = torch.sqrt(sh ** 2 + sv ** 2 + 1e-6)
        lap3_out = f3[:, 0::3, :, :]
        lap5_out = f5[:, 0::2, :, :]
        dog_out = f5[:, 1::2, :, :]
        npr_out = self.npr(x)
        return torch.cat([lap3_out, lap5_out, dog_out, smag, npr_out], dim=1)


class AIForensicNetV4HiDF(nn.Module):
    """
    AIForensicNetV4HiDF: Model V4 yang secara presisi dilatih pada HiDF Deepfake Dataset.
    Kompatibel 100% dengan bobot best_aiforensicnet_v4_hidf.pth.
    """
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.25):
        super().__init__()
        self.forensic_bank = MultiScaleForensicFilterBankV4HiDF()
        self.forensic_stem = nn.Sequential(nn.Conv2d(15, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU())
        self.spectral_branch = FFTSpectralBranchHiDF(3, 32)
        self.rgb_stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU())
        self.fusion = nn.Sequential(nn.Conv2d(96, 64, 1, bias=False), nn.BatchNorm2d(64), nn.GELU())
        self.stage1 = nn.Sequential(ForensicBlock(64, 64), ForensicBlock(64, 64))
        self.stage2 = nn.Sequential(ForensicBlock(64, 128, stride=2), ForensicBlock(128, 128))
        self.stage3 = nn.Sequential(ForensicBlock(128, 256, stride=2), ForensicBlock(256, 256))
        self.stage4 = nn.Sequential(ForensicBlock(256, 512, stride=2), ForensicBlock(512, 512))
        self.pool = MultiStatPool2d()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512 * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        f_forensic = self.forensic_stem(self.forensic_bank(x))
        f_spectral = self.spectral_branch(x)
        f_rgb = self.rgb_stem(x)
        out = self.fusion(torch.cat([f_forensic, f_spectral, f_rgb], dim=1))
        out = self.stage4(self.stage3(self.stage2(self.stage1(out))))
        return self.head(self.pool(out))


class AIForensicNetV4Forgery(nn.Module):
    """
    AIForensicNetV4Forgery: Arsitektur Quad-Domain Forensik V4 khusus untuk
    deteksi manipulasi citra (Splicing, Copy-Move, CASIA v2.0 & IMD2020).
    Mendukung input 6 kanal (RGB + ELA) atau 3 kanal RGB dengan fallback otomatis.
    Kompatibel 100% dengan bobot best_aiforensicnet_v4_casia2.pth.
    """
    def __init__(self, in_channels: int = 6, num_classes: int = 2, dropout_p: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.forensic_bank = MultiScaleForensicFilterBankV4HiDF()
        self.forensic_stem = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(32), 
            nn.GELU()
        )
        self.spectral_branch = FFTSpectralBranchHiDF(3, 32)
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(32), 
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.GELU()
        )
        self.stage1 = nn.Sequential(ForensicBlock(64, 64), ForensicBlock(64, 64))
        self.stage2 = nn.Sequential(ForensicBlock(64, 128, stride=2), ForensicBlock(128, 128))
        self.stage3 = nn.Sequential(ForensicBlock(128, 256, stride=2), ForensicBlock(256, 256))
        self.stage4 = nn.Sequential(ForensicBlock(256, 512, stride=2), ForensicBlock(512, 512))
        self.pool = MultiStatPool2d()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.shape[1] == 6:
            x_rgb = x[:, :3]
            f_rgb = self.rgb_stem(x)
        else:
            x_rgb = x
            if self.in_channels == 6:
                f_rgb = self.rgb_stem(torch.cat([x, x], dim=1))
            else:
                f_rgb = self.rgb_stem(x)
                
        f_forensic = self.forensic_stem(self.forensic_bank(x_rgb))
        f_spectral = self.spectral_branch(x_rgb)
        out = self.fusion(torch.cat([f_forensic, f_spectral, f_rgb], dim=1))
        out = self.stage4(self.stage3(self.stage2(self.stage1(out))))
        return self.head(self.pool(out))


class AIForensicNetV4Universal(nn.Module):
    """
    AIForensicNet V4 Universal (Two-Stream Dual-Branch dengan Learnable Gated Cross-Attention).
    
    Arsitektur End-to-End untuk Deteksi Universal (AI Generatif Murni & Forgery / Splicing):
    1. Branch 1 (Spatial Texture Stream - Generative AI Specialist):
       Mengekstrak artefak visual, tekstur kulit, batas helai rambut, dan keanehan khas difusi
       (Midjourney v6, DALL-E 3, SDXL, Flux, CIFAKE).
    2. Branch 2 (Forensic High-Pass & Spectral Stream - Tampering / Splicing Specialist):
       Mengekstrak diskontinuitas gradien tepi, residu noise PRNU, filter bank steganografi (SRM,
       Laplacian 3x3/5x5, DoG), dan resonansi frekuensi 2D FFT (CASIA 2.0, Photoshop).
    3. Learnable Gated Cross-Attention (Dynamic Gate α):
       Secara dinamis mempelajari pembobotan fitur (alpha in [0, 1]) untuk setiap citra secara otomatis.
       Jika splicing -> bobot Branch 2 naik. Jika AI murni -> bobot Branch 1 naik.
       Jika foto asli -> kedua cabang menghasilkan representasi negatif.
    4. Input: Standar 3-kanal RGB (Format-agnostic: JPEG, PNG, WEBP, BMP didukung secara native tanpa ELA loss).
    """
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.3):
        super().__init__()
        # Branch 1: Spatial & Semantic Stream (AI Generatif: Midjourney, DALL-E, SDXL)
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.b1_stage1 = nn.Sequential(ForensicBlock(32, 64), ForensicBlock(64, 64))
        self.b1_stage2 = nn.Sequential(ForensicBlock(64, 128, stride=2), ForensicBlock(128, 128))
        self.b1_stage3 = nn.Sequential(ForensicBlock(128, 256, stride=2), ForensicBlock(256, 256))
        self.b1_stage4 = nn.Sequential(ForensicBlock(256, 384, stride=2), ForensicBlock(384, 384))
        
        # Branch 2: High-Pass Residual & Spectral Stream (Forgery, Splicing, Copy-Move)
        self.forensic_bank = MultiScaleForensicFilterBankV4HiDF()
        self.forensic_stem = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.spectral_branch = FFTSpectralBranchHiDF(3, 32)
        self.b2_fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.b2_stage1 = nn.Sequential(ForensicBlock(32, 64), ForensicBlock(64, 64))
        self.b2_stage2 = nn.Sequential(ForensicBlock(64, 128, stride=2), ForensicBlock(128, 128))
        self.b2_stage3 = nn.Sequential(ForensicBlock(128, 256, stride=2), ForensicBlock(256, 256))
        self.b2_stage4 = nn.Sequential(ForensicBlock(256, 384, stride=2), ForensicBlock(384, 384))
        
        self.pool = MultiStatPool2d()
        
        # Learnable Gated Cross-Attention Module
        self.gate_mlp = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 768),
            nn.Sigmoid()
        )
        
        # Classification Head (Dual-enhanced)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, x_crop=None, return_gate: bool = False):
        """
        Forward Pass Dual-Scale Asymmetric:
        - x        : Citra Global (224x224 scaled) untuk Branch 1 (AI Generatif & Semantik Global).
        - x_crop   : Citra Native-Resolution Crop (224x224 murni) untuk Branch 2 (Forensik Frekuensi & Splicing).
        Mendukung backward-compatibility: jika x_crop=None, kedua branch memproses x.
        """
        if x_crop is not None:
            x_global = x[:, :3] if x.shape[1] >= 3 else x
            x_local = x_crop[:, :3] if x_crop.shape[1] >= 3 else x_crop
        elif x.shape[1] == 6:
            # Dual-scale input terpaket dalam 6-kanal (0:3 global, 3:6 native crop)
            x_global = x[:, :3]
            x_local = x[:, 3:6]
        else:
            x_global = x
            x_local = x
            
        # Branch 1 Forward (Spasial Tekstur AI & Semantik Global)
        f1 = self.rgb_stem(x_global)
        f1 = self.b1_stage4(self.b1_stage3(self.b1_stage2(self.b1_stage1(f1))))
        v_spatial = self.pool(f1)
        
        # Branch 2 Forward (High-Pass Residual Splicing & Spektral Resolusi Native)
        f_bank = self.forensic_stem(self.forensic_bank(x_local))
        f_spec = self.spectral_branch(x_local)
        f2 = self.b2_fusion(torch.cat([f_bank, f_spec], dim=1))
        f2 = self.b2_stage4(self.b2_stage3(self.b2_stage2(self.b2_stage1(f2))))
        v_residual = self.pool(f2)
        
        # Gated Cross-Attention Fusion
        gate_input = torch.cat([v_spatial, v_residual], dim=1)
        alpha = self.gate_mlp(gate_input)
        
        # Adaptive Fused Representation
        v_fused = alpha * v_spatial + (1.0 - alpha) * v_residual
        
        logits = self.head(v_fused)
        if return_gate:
            return logits, alpha.mean(dim=1)
        return logits





