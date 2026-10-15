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
