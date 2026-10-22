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
        self.pool = MeanStdPool2d()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes)
        )

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


