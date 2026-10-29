"""
AIFusionNet-Universal (V5 Architecture)
======================================
Arsitektur Forensik Citra Mutakhir yang Dirancang Khusus dari Awal (From Scratch, Tanpa Pretrained):
1. Dual-Domain Multi-Scale Input:
   - Stream 1 (RGB Generative Stream): Menangkap tekstur semantik, difusi mikro, dan artefak AI synthesis.
   - Stream 2 (Forensic Residual Stream): Menangkap 15-kanal SRM (Spatial Rich Model) + ELA (Error Level Analysis)
     + 2D FFT Spektral untuk mendeteksi splicing, copy-move, inpainting, dan phase anomaly.
2. Modern Inverted-Bottleneck ConvNeXt Stages dengan Depthwise 7x7 Convolutions.
3. Squeeze-and-Excitation (SE) Channel Attention di setiap stage.
4. Multi-Stat Pooling (Mean + Std + Max) untuk menangkap varians noise dan anomali lokal.
5. Learnable Cross-Attention Fusion dengan Auxiliary Supervision untuk mencegah branch collapse.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. LAYER NORMALIZATION CHANNELS FIRST
# ==============================================================================

class LayerNorm2d(nn.Module):
    """LayerNorm untuk representasi citra format (B, C, H, W)."""
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


# ==============================================================================
# 2. SQUEEZE-AND-EXCITATION (SE) ATTENTION
# ==============================================================================

class SqueezeExcitation(nn.Module):
    """SE Attention untuk adaptasi bobot kanal fitur forensik."""
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
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ==============================================================================
# 3. MODERN DEPTHWISE CONVOLUTION BLOCK (ConvNeXt-Style)
# ==============================================================================

class ModernConvBlock(nn.Module):
    """
    Blok Konvolusi Modern (Depthwise 7x7 -> LayerNorm -> Pointwise 4x -> GELU -> SE -> Pointwise 1x)
    Sangat stabil saat dilatih dari nol (from scratch) tanpa pretrained weights.
    """
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.se = SqueezeExcitation(dim)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.se(x)
        return residual + x


# ==============================================================================
# 4. STATIC FORENSIC FILTER BANK (SRM + LAPLACIAN + HIGH-PASS)
# ==============================================================================

class StaticForensicFilterBank(nn.Module):
    """
    Kumpulan filter steganografi & residu frekuensi tinggi tetap (non-trainable):
    1. 1st Order Residuals (Horizontal, Vertikal, Diagonal)
    2. 2nd Order Residuals (Laplacian 3x3, 5x5)
    3. Spatial Rich Model (SRM) Filters: KV, MinMax, Edge
    Output: 12 kanal fitur residu (4 filter x 3 kanal RGB).
    """
    def __init__(self):
        super().__init__()
        # 4 filter dasar forensik berukuran 5x5
        f1 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  2, -4,  2,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=torch.float32) / 4.0  # KV Kernel

        f2 = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 12.0 # 5x5 Laplacian Edge

        f3 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0, -1,  4, -1,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=torch.float32) / 4.0  # 3x3 Standard Laplacian

        f4 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  0,  1,  0],
            [ 0, -2,  0,  2,  0],
            [ 0, -1,  0,  1,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=torch.float32) / 4.0  # Sobel Horizontal Gradient

        filters = torch.stack([f1, f2, f3, f4], dim=0)  # (4, 5, 5)
        # Replikasi untuk tiap kanal RGB (4 filter x 3 RGB = 12 kanal output)
        weight = torch.zeros(12, 1, 5, 5)
        for i in range(4):
            for c in range(3):
                weight[i * 3 + c, 0] = filters[i]

        self.conv = nn.Conv2d(3, 12, kernel_size=5, padding=2, bias=False, groups=3)
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.conv(x))


# ==============================================================================
# 5. 2D FFT RADIAL SPECTRAL RESIDUAL MODULE
# ==============================================================================

class Spectral2DResidualModule(nn.Module):
    """
    Mengekstrak anomali frekuensi spektral 2D FFT:
    Mendeteksi artefak grid/checkerboard dan resonansi frekuensi tinggi khas model difusi AI.
    Output: 3 kanal representasi spektral ter-skala.
    """
    def __init__(self):
        super().__init__()
        self.norm = LayerNorm2d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast Fourier Transform 2D
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        log_spec = torch.log1p(magnitude)
        # Normalisasi spektrum ke rentang dinamis stabil
        return self.norm(log_spec)


# ==============================================================================
# 6. MULTI-STATISTICAL POOLING (Mean + Std + Max)
# ==============================================================================

class MultiStatPool2d(nn.Module):
    """
    Menggabungkan tiga jenis statistik spasial:
    - Global Average: Karakteristik rata-rata tekstur citra
    - Global Standard Deviation: Varians dan fluktuasi noise sensor
    - Global Max: Anomali titik manipulasi lokal (misal tepi splicing)
    Output: Vektor dimensi C * 3.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        mean = torch.mean(x, dim=[2, 3])
        var = torch.var(x, dim=[2, 3], unbiased=False)
        std = torch.sqrt(var + self.eps)
        max_val = torch.amax(x, dim=[2, 3])
        return torch.cat([mean, std, max_val], dim=1)


# ==============================================================================
# 7. BACKBONE STAGES (ConvNeXt-Style Hierarchy)
# ==============================================================================

def make_stage(in_dim: int, out_dim: int, depth: int, stride: int = 2) -> nn.Sequential:
    layers = []
    if stride == 2:
        layers.append(LayerNorm2d(in_dim))
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2))
    elif in_dim != out_dim:
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
    for _ in range(depth):
        layers.append(ModernConvBlock(out_dim))
    return nn.Sequential(*layers)


# ==============================================================================
# 8. CORE ARCHITECTURE: AIFusionNet-Universal (V5)
# ==============================================================================

class AIFusionNetUniversal(nn.Module):
    """
    Dual-Stream Modern ConvNet dengan Learnable Gated Cross-Attention:
    - Branch 1 (RGB Generative Stream):
      Input: RGB 3-kanal.
      Fokus: Semantik visual, artefak kulit difusi, dan tekstur render AI.
    - Branch 2 (Forensic Residual Stream):
      Input: 15-kanal (12-ch Static Filter Bank + 3-ch 2D FFT Spektral).
      Fokus: Residu steganografi, ketidakkonsistenan noise kamera, splicing, and copy-move.
    - Multi-Stat Pooling & Cross-Attention Head:
      Menggabungkan representasi kedua branch dengan pembobotan adaptif.
    - Auxiliary Classifiers:
      Memastikan kedua branch terlatih secara optimal tanpa ada yang terabaikan.
    """
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.3):
        super().__init__()

        # --- STREAM 1: RGB Generative Branch ---
        self.b1_stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=4, stride=4),
            LayerNorm2d(48)
        )
        self.b1_stage1 = make_stage(48, 48, depth=2, stride=1)
        self.b1_stage2 = make_stage(48, 96, depth=2, stride=2)
        self.b1_stage3 = make_stage(96, 192, depth=3, stride=2)
        self.b1_stage4 = make_stage(192, 384, depth=2, stride=2)

        # --- STREAM 2: Forensic Residual & Spectral Branch ---
        self.forensic_bank = StaticForensicFilterBank()
        self.spectral_module = Spectral2DResidualModule()
        # Input 12 (filter bank) + 3 (spectral) = 15 kanal
        self.b2_stem = nn.Sequential(
            nn.Conv2d(15, 48, kernel_size=4, stride=4),
            LayerNorm2d(48)
        )
        self.b2_stage1 = make_stage(48, 48, depth=2, stride=1)
        self.b2_stage2 = make_stage(48, 96, depth=2, stride=2)
        self.b2_stage3 = make_stage(96, 192, depth=3, stride=2)
        self.b2_stage4 = make_stage(192, 384, depth=2, stride=2)

        # Multi-Stat Pooling: 384 kanal -> 384 * 3 = 1152 dimensi
        self.pool = MultiStatPool2d()
        feat_dim = 384 * 3  # 1152

        # --- GATED CROSS-ATTENTION MODULE ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, feat_dim),
            nn.Sigmoid()
        )

        # --- CLASSIFICATION HEADS ---
        # Main Head (Fused Representation)
        self.main_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        # Auxiliary Head 1: RGB Spatial Only
        self.aux_head_rgb = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

        # Auxiliary Head 2: Forensic Residual Only
        self.aux_head_forensic = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x_global: torch.Tensor, x_patch: torch.Tensor = None):
        """
        Mengekstrak fitur Dual-Scale:
        - x_global: Citra keseluruhan beresolusi 224x224 (Global Semantics, Anatomi, Pencahayaan)
        - x_patch : Patch 224x224 pada resolusi murni 1:1 tanpa resize (Micro-Forensic Residuals SRM + FFT)
        Jika x_patch bernilai None, model secara otomatis memakai x_global (backward compatible).
        """
        if x_patch is None:
            x_patch = x_global

        # Branch 1 (RGB Global Semantic Stream)
        f1 = self.b1_stem(x_global)
        f1 = self.b1_stage4(self.b1_stage3(self.b1_stage2(self.b1_stage1(f1))))
        v_rgb = self.pool(f1)

        # Branch 2 (Residu Forensik Mikro + Spektral dari Patch Resolusi Murni)
        res_bank = self.forensic_bank(x_patch)
        res_spec = self.spectral_module(x_patch)
        res_input = torch.cat([res_bank, res_spec], dim=1)
        f2 = self.b2_stem(res_input)
        f2 = self.b2_stage4(self.b2_stage3(self.b2_stage2(self.b2_stage1(f2))))
        v_forensic = self.pool(f2)

        return v_rgb, v_forensic

    def forward(self, x_global: torch.Tensor, x_patch: torch.Tensor = None,
                return_all: bool = False, return_gate: bool = False):
        """
        Forward Pass AIFusionNet Universal V5 Dual-Scale:
        - x_global: Citra global (B, 3, 224, 224)
        - x_patch : Patch resolusi asli (B, 3, 224, 224), opsional (default = x_global)
        - return_all: Mengembalikan main_logits beserta auxiliary logits
        - return_gate: Mengembalikan bobot fusi alpha (atensi kontribusi RGB vs Forensik)
        """
        v_rgb, v_forensic = self.extract_features(x_global, x_patch)

        # Fusi Atensi Gated
        gate_input = torch.cat([v_rgb, v_forensic], dim=1)
        alpha = self.fusion_gate(gate_input)  # Rentang (0.0 s.d 1.0)
        v_fused = alpha * v_rgb + (1.0 - alpha) * v_forensic

        main_logits = self.main_head(v_fused)

        if return_all:
            aux_rgb = self.aux_head_rgb(v_rgb)
            aux_forensic = self.aux_head_forensic(v_forensic)
            return main_logits, aux_rgb, aux_forensic, alpha.mean(dim=1)

        if return_gate:
            return main_logits, alpha.mean(dim=1)

        return main_logits


def build_v5_model(num_classes: int = 2, dropout_p: float = 0.3) -> AIFusionNetUniversal:
    """Helper pembuat model AIFusionNet Universal V5."""
    return AIFusionNetUniversal(num_classes=num_classes, dropout_p=dropout_p)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_v5_model().to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    out, aux1, aux2, gate = model(dummy_input, return_all=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== AIFusionNet-Universal V5 Architecture ===")
    print(f"Total Parameter     : {total_params:,} ({total_params * 4 / (1024**2):.2f} MB)")
    print(f"Trainable Parameter : {trainable_params:,}")
    print(f"Main Logits Shape   : {out.shape}")
    print(f"Aux RGB Logits      : {aux1.shape}")
    print(f"Aux Forensic Logits : {aux2.shape}")
    print(f"Mean Gate Alpha     : {gate.tolist()}")
