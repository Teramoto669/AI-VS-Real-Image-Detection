import warnings
warnings.filterwarnings('ignore', message='.*Palette images with Transparency.*')
warnings.filterwarnings('ignore', message='.*xFormers is not available.*')
import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Konfigurasi Halaman
st.set_page_config(
    page_title="AI vs Real Image Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Minimalis, utilitarian, enhanced dropzone & paste support
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container constraints */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header styling */
    .app-header {
        margin-bottom: 1.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.25rem;
    }

    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
    }

    .app-desc {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 0.35rem;
    }

    /* Drag & Drop Target Area - Rata Kiri & Tinggi Presisi */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        background-color: #121826 !important;
        padding: 0.85rem 1.25rem !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
        min-height: 82px !important;
        height: 82px !important;
        box-sizing: border-box !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploaderDropzone"]:focus-within {
        border-color: #3B82F6 !important;
        background-color: #162035 !important;
    }

    [data-testid="stFileUploaderDropzone"] > div {
        text-align: left !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 1rem !important;
        width: 100% !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        text-align: left !important;
        align-items: flex-start !important;
        justify-content: center !important;
        display: flex !important;
        flex-direction: column !important;
        color: #94A3B8 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] div {
        text-align: left !important;
        display: block !important;
    }

    /* Result Panel */
    .verdict-card {
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .verdict-ai {
        background: rgba(244, 63, 94, 0.08);
        border: 1px solid rgba(244, 63, 94, 0.3);
    }

    .verdict-real {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .verdict-modified {
        background: rgba(245, 158, 11, 0.09);
        border: 1px solid rgba(245, 158, 11, 0.38);
    }

    .verdict-tag {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 0.35rem;
    }

    .verdict-label {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.35;
        margin: 0;
    }

    .label-ai {
        color: #FB7185;
    }

    .label-real {
        color: #34D399;
    }

    .label-modified {
        color: #FBBF24;
        font-size: 1.15rem;
    }

    /* Stat Box */
    .stat-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }

    .stat-card {
        background: #121826;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }

    .stat-title {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .stat-num {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F1F5F9;
        font-variant-numeric: tabular-nums;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    /* Metadata strip under image */
    .image-meta-bar {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.5rem;
        padding: 0.4rem 0.6rem;
        background: #121826;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        display: flex;
        justify-content: space-between;
    }

    /* Sidebar styling */
    .sidebar-spec-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.5rem 0.85rem;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .spec-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.82rem;
        line-height: 1.35;
    }

    .spec-row:last-child {
        border-bottom: none;
    }

    .spec-key {
        color: #94A3B8;
        font-weight: 500;
        flex-shrink: 0;
        min-width: 76px;
    }

    .spec-val {
        color: #F1F5F9;
        font-weight: 600;
        text-align: right;
        flex-grow: 1;
        word-break: normal;
        overflow-wrap: break-word;
    }

    .spec-val-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: #34D399;
        font-weight: 600;
    }

    .spec-val-badge-wait {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: #F87171;
        font-weight: 600;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        display: inline-block;
        box-shadow: 0 0 6px rgba(16, 185, 129, 0.5);
    }

    .status-dot-wait {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #EF4444;
        display: inline-block;
        box-shadow: 0 0 6px rgba(239, 68, 68, 0.5);
    }

    .spec-desc-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        font-size: 0.8rem;
        color: #94A3B8;
        line-height: 1.5;
    }

    .spec-desc-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

MAX_FILE_SIZE_MB = 10.0
MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)

# Pipeline Transformasi
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODELS_CONFIG = {
    "DINOv2-Small (Meta AI)": {
        "paths": ["best_dinov2_vits14_ai_vs_real.pth", "backup/best_dinov2_vits14_ai_vs_real.pth"],
        "arch": "dinov2_vits14",
        "name": "DINOv2-Small",
        "arch_name": "Vision Transformer (ViT)",
        "params": "22.1 Juta (Meta AI)",
        "desc": "Self-supervised ViT Meta AI dengan deteksi patch-level yang sangat akurat.",
    },
    "ConvNeXt-Tiny (Meta AI)": {
        "paths": ["best_convnext_tiny_ai_vs_real.pth", "backup/best_convnext_tiny_ai_vs_real.pth"],
        "arch": "convnext_tiny",
        "name": "ConvNeXt-Tiny",
        "arch_name": "Modern ConvNet",
        "params": "28 Juta (Meta AI)",
        "desc": "Arsitektur ConvNet modern Meta AI yang sangat peka terhadap artefak mikro AI.",
    },
    "EfficientNetV2-S (Google)": {
        "paths": ["best_efficientnet_v2_s_ai_vs_real.pth", "backup/best_efficientnet_v2_s_ai_vs_real.pth"],
        "arch": "efficientnet_v2_s",
        "name": "EfficientNetV2-S",
        "arch_name": "Fused-MBConv",
        "params": "21.5 Juta (Google)",
        "desc": "Arsitektur Google dengan Fused-MBConv yang cepat dan sangat efisien.",
    },
    "AIForensicNet (Custom Net)": {
        "paths": ["best_aiforensicnet_ai_vs_real.pth", "backup/best_aiforensicnet_ai_vs_real.pth"],
        "arch": "aiforensicnet",
        "name": "AIForensicNet",
        "arch_name": "Dual-Stream Forensic CNN",
        "params": "2.85 Juta (From Scratch)",
        "desc": "Arsitektur kustom dual-stream (Laplacian High-Pass + Spatial RGB) dengan Squeeze-and-Excitation attention.",
    },
    "AIForensicNet V2 (Optimized)": {
        "paths": ["best_aiforensicnet_v2_ai_vs_real.pth", "backup/best_aiforensicnet_v2_ai_vs_real.pth"],
        "arch": "aiforensicnet_v2",
        "name": "AIForensicNet V2",
        "arch_name": "Multi-Filter Dual-Attention CNN",
        "params": "3.52 Juta (From Scratch)",
        "desc": "Arsitektur kustom teroptimasi: Multi-Filter SRM + Bayar Constrained Conv + Dual Attention + Concat Pooling.",
    },
    "AIForensicNet V3 (Multi-Scale)": {
        "paths": ["best_aiforensicnet_v3_ai_vs_real.pth", "backup/best_aiforensicnet_v3_ai_vs_real.pth"],
        "arch": "aiforensicnet_v3",
        "name": "AIForensicNet V3",
        "arch_name": "Multi-Scale Static Bank + Mean-Std Pooling",
        "params": "3.07 Juta (From Scratch)",
        "desc": "Arsitektur kustom generasi ke-3: Multi-Scale Static Filter Bank (12-ch) + Dual-Stream + Mean-Std Pooling.",
    },
    "AIForensicNet V4 (CommunityForensics)": {
        "paths": ["best_aiforensicnet_v4_cf.pth", "backup/best_aiforensicnet_v4_cf.pth"],
        "arch": "aiforensicnet_v4",
        "name": "AIForensicNet V4 (CF)",
        "arch_name": "Tri-Stream (NPR + 2D-FFT + RGB)",
        "params": "2.93 Juta (OwensLab/CommunityForensics)",
        "desc": "Arsitektur kustom generasi ke-4 dilatih pada dataset SOTA CommunityForensics (CVPR 2025) dengan Multi-Crop Native Resolution.",
    },
    "AIForensicNet V4 (Thunderbeast)": {
        "paths": ["best_aiforensicnet_v4_ai_vs_real.pth", "backup/best_aiforensicnet_v4_ai_vs_real.pth"],
        "arch": "aiforensicnet_v4",
        "name": "AIForensicNet V4 (TB)",
        "arch_name": "Tri-Stream (NPR + 2D-FFT + RGB)",
        "params": "2.93 Juta (Thunderbeast Dataset)",
        "desc": "Arsitektur kustom generasi ke-4 dilatih pada dataset Thunderbeast dengan Multi-Crop Native Resolution.",
    },
}

# Deteksi model yang tersedia di penyimpanan lokal
AVAILABLE_MODELS = {}
for label, cfg in MODELS_CONFIG.items():
    for p in cfg["paths"]:
        if os.path.exists(p):
            AVAILABLE_MODELS[label] = {**cfg, "active_path": p}
            break


@st.cache_resource(show_spinner=False)
def get_model(model_path: str, arch: str):
    """Memuat model secara dinamis (DINOv2-Small, ConvNeXt-Tiny, EfficientNetV2-S, atau AIForensicNet V1-V4)."""
    if not os.path.exists(model_path):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch == "dinov2_vits14":
        class DINOv2AppClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
                    warnings.filterwarnings("ignore", category=UserWarning)
                    hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
                    if os.path.exists(hub_dir):
                        self.backbone = torch.hub.load(hub_dir, "dinov2_vits14", source="local")
                    else:
                        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(self.backbone.embed_dim, 2)
                )

            def forward(self, x):
                return self.classifier(self.backbone(x))

        model = DINOv2AppClassifier()
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 2)
        )
    elif arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 2)
        )
    elif arch == "aiforensicnet":
        from model_custom import AIForensicNet
        model = AIForensicNet(num_classes=2)
    elif arch == "aiforensicnet_v2":
        from model_custom import AIForensicNetV2
        model = AIForensicNetV2(num_classes=2)
    elif arch == "aiforensicnet_v3":
        from model_custom import AIForensicNetV3
        model = AIForensicNetV3(num_classes=2)
    elif arch == "aiforensicnet_v4":
        from model_custom import AIForensicNetV4
        model = AIForensicNetV4(num_classes=2)
    else:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 2)
        )

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device


def run_inference(image: Image.Image, model: nn.Module, device: torch.device, arch: str = ""):
    """
    Menjalankan inferensi cerdas adaptif berdasarkan karakteristik arsitektur:
    - Untuk model forensik (AIForensicNet V2, V3, V4): Menggunakan Multi-Crop Native Resolution TTA
      (Center Crop + 4 Corner Crops + Global Resized Context + Flip) untuk menjaga kestabilan
      frekuensi tinggi dan noise sensor kamera asli tanpa terdistorsi downsampling.
    - Untuk model semantik (DINOv2, ConvNeXt, EfficientNet): Menggunakan Global Resized TTA.
    """
    if image.mode == 'P' and 'transparency' in image.info:
        image = image.convert('RGBA')
    rgb_image = image.convert("RGB")
    w, h = rgb_image.size

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if arch in ["aiforensicnet_v2", "aiforensicnet_v3", "aiforensicnet_v4"]:
        crop_size = 224
        # Pastikan dimensi minimal memenuhi ukuran crop
        if min(w, h) < crop_size:
            scale = float(crop_size) / float(min(w, h))
            proc_img = rgb_image.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BILINEAR)
            pw, ph = proc_img.size
        else:
            proc_img = rgb_image
            pw, ph = w, h

        # 5 Patch resolusi native 1:1 menjaga jejak noise PRNU sensor kamera
        cx, cy = (pw - crop_size) // 2, (ph - crop_size) // 2
        patches = [
            proc_img.crop((cx, cy, cx + crop_size, cy + crop_size)), # Center
            proc_img.crop((0, 0, crop_size, crop_size)),             # Top-Left
            proc_img.crop((pw - crop_size, 0, pw, crop_size)),       # Top-Right
            proc_img.crop((0, ph - crop_size, crop_size, ph)),       # Bottom-Left
            proc_img.crop((pw - crop_size, ph - crop_size, pw, ph)), # Bottom-Right
            rgb_image.resize((crop_size, crop_size), Image.Resampling.BILINEAR), # Global layout context
        ]

        tensor_list = []
        for p in patches:
            t = norm_transform(p)
            tensor_list.append(t)
            tensor_list.append(torch.flip(t, dims=[2])) # horizontal flip TTA

        batch_tensors = torch.stack(tensor_list).to(device)

        with torch.no_grad():
            outputs = model(batch_tensors)
            probs = torch.softmax(outputs, dim=1)
            mean_probs = probs.mean(dim=0)
            confidence, predicted_idx = torch.max(mean_probs, 0)

        ai_prob = mean_probs[0].item() * 100.0
        real_prob = mean_probs[1].item() * 100.0
    else:
        tensor_img = transform_pipeline(rgb_image).unsqueeze(0).to(device)
        tensor_img_flipped = torch.flip(tensor_img, dims=[3])

        with torch.no_grad():
            outputs_orig = model(tensor_img)
            outputs_flip = model(tensor_img_flipped)
            probs_orig = torch.softmax(outputs_orig, dim=1)[0]
            probs_flip = torch.softmax(outputs_flip, dim=1)[0]
            probs = (probs_orig + probs_flip) / 2.0
            confidence, predicted_idx = torch.max(probs, 0)

        ai_prob = probs[0].item() * 100.0
        real_prob = probs[1].item() * 100.0

    return predicted_idx.item(), confidence.item() * 100.0, ai_prob, real_prob


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.25rem;">
        Spesifikasi Model
    </div>
    <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.85rem;">
        Konfigurasi & status inferensi
    </div>
    """, unsafe_allow_html=True)

    if len(AVAILABLE_MODELS) > 1:
        selected_model_label = st.selectbox(
            "Pilih Model Deteksi",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            help="Pilih model yang ingin digunakan untuk deteksi citra."
        )
        active_cfg = AVAILABLE_MODELS[selected_model_label]
        model, device = get_model(active_cfg["active_path"], active_cfg["arch"])
    elif len(AVAILABLE_MODELS) == 1:
        selected_model_label = list(AVAILABLE_MODELS.keys())[0]
        active_cfg = AVAILABLE_MODELS[selected_model_label]
        model, device = get_model(active_cfg["active_path"], active_cfg["arch"])
    else:
        # Default fallback jika belum ada checkpoint
        active_cfg = list(MODELS_CONFIG.values())[0]
        model, device = None, None

    device_label = "CUDA" if (device and device.type == "cuda") else "CPU"
    status_html = (
        '<span class="spec-val-badge"><span class="status-dot"></span>Siap</span>'
        if model else
        '<span class="spec-val-badge-wait"><span class="status-dot-wait"></span>Menunggu</span>'
    )
    eval_mode_label = (
        "Multi-Crop Native TTA"
        if active_cfg.get("arch", "") in ["aiforensicnet_v2", "aiforensicnet_v3", "aiforensicnet_v4"]
        else "Global Resized TTA"
    )

    st.markdown(f"""
    <div class="sidebar-spec-card">
        <div class="spec-row"><span class="spec-key">Status</span><span class="spec-val">{status_html}</span></div>
        <div class="spec-row"><span class="spec-key">Model</span><span class="spec-val">{active_cfg['name']}</span></div>
        <div class="spec-row"><span class="spec-key">Arsitektur</span><span class="spec-val">{active_cfg['arch_name']}</span></div>
        <div class="spec-row"><span class="spec-key">Parameter</span><span class="spec-val">{active_cfg['params']}</span></div>
        <div class="spec-row"><span class="spec-key">Perangkat</span><span class="spec-val">{device_label}</span></div>
        <div class="spec-row"><span class="spec-key">Mode Evaluasi</span><span class="spec-val">{eval_mode_label}</span></div>
        <div class="spec-row"><span class="spec-key">Input Res</span><span class="spec-val">224 × 224 px</span></div>
        <div class="spec-row"><span class="spec-key">Batas File</span><span class="spec-val">Maks. {int(MAX_FILE_SIZE_MB)} MB</span></div>
    </div>
    <div class="spec-desc-box">
        <div class="spec-desc-label">Keterangan Model</div>
        {active_cfg['desc']}
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown("""
<div class="app-header">
    <h1 class="app-title">AI vs Real Image Detector</h1>
    <div class="app-desc">Klasifikasi citra untuk membedakan gambar sintetis (AI-Generated) dan foto asli (Real Image).</div>
</div>
""", unsafe_allow_html=True)

# Area Input: File Uploader (Drag & Drop) dan Tombol Paste Clipboard
col_upload, col_paste = st.columns([6, 1], gap="small")

with col_upload:
    uploaded_file = st.file_uploader(
        "Pilih file gambar",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help=f"Batas ukuran maksimal {int(MAX_FILE_SIZE_MB)} MB (format JPG, PNG, WEBP)"
    )

with col_paste:
    # Menggunakan components.html untuk menyisipkan HTML/JS paste & drop button
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body {
        background: transparent;
        overflow: hidden;
        height: 82px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .paste-btn {
        width: 100%;
        height: 82px;
        background-color: #121826;
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: #94A3B8;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        outline: none;
      }
      .paste-btn:hover {
        border-color: #3B82F6;
        background-color: #162035;
        color: #F8FAFC;
      }
      .paste-btn:active {
        transform: scale(0.96);
      }
      .paste-btn svg {
        width: 25px;
        height: 25px;
        stroke: currentColor;
        stroke-width: 1.9;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
      }
    </style>
    </head>
    <body>
      <button class="paste-btn" id="pasteActionBtn" title="Paste dari clipboard (atau tekan Ctrl + V)">
        <svg viewBox="0 0 24 24">
          <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1 2-2h2"></path>
          <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
          <path d="M9 12h6"></path>
          <path d="M9 16h6"></path>
        </svg>
      </button>
      <script>
        (function() {
          function setupListeners() {
            try {
              const parentDoc = window.parent.document;
              if (!parentDoc) return;

              function getFileInput() {
                return parentDoc.querySelector('section[data-testid="stFileUploaderDropzone"] input[type="file"]') ||
                       parentDoc.querySelector('input[type="file"]');
              }

              if (!parentDoc._imageListenersAttached) {
                parentDoc._imageListenersAttached = true;

                // Global Ctrl+V handler
                parentDoc.addEventListener('paste', function(e) {
                  const items = (e.clipboardData || window.clipboardData)?.items;
                  if (!items) return;
                  for (let i = 0; i < items.length; i++) {
                    if (items[i].type.indexOf('image') !== -1) {
                      const file = items[i].getAsFile();
                      if (file) {
                        const input = getFileInput();
                        if (input) {
                          const dt = new DataTransfer();
                          dt.items.add(file);
                          input.files = dt.files;
                          input.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                      }
                      break;
                    }
                  }
                });

                // Global Drag & Drop handler
                parentDoc.addEventListener('dragover', function(e) { e.preventDefault(); });
                parentDoc.addEventListener('drop', function(e) {
                  const dt = e.dataTransfer;
                  if (dt && dt.files && dt.files.length > 0) {
                    const file = dt.files[0];
                    if (file.type && file.type.startsWith('image/')) {
                      const input = getFileInput();
                      if (input && e.target !== input) {
                        e.preventDefault();
                        const newDt = new DataTransfer();
                        newDt.items.add(file);
                        input.files = newDt.files;
                        input.dispatchEvent(new Event('change', { bubbles: true }));
                      }
                    }
                  }
                });
              }

              // Click handler untuk tombol Paste
              const btn = document.getElementById('pasteActionBtn');
              if (btn && !btn._hasClick) {
                btn._hasClick = true;
                btn.addEventListener('click', async function() {
                  try {
                    const nav = window.parent.navigator || navigator;
                    const items = await nav.clipboard.read();
                    for (let i = 0; i < items.length; i++) {
                      const item = items[i];
                      for (let j = 0; j < item.types.length; j++) {
                        const type = item.types[j];
                        if (type.startsWith('image/')) {
                          const blob = await item.getType(type);
                          const ext = type.split('/')[1] || 'png';
                          const file = new File([blob], 'clipboard_image.' + ext, { type: type });
                          const input = getFileInput();
                          if (input) {
                            const dt = new DataTransfer();
                            dt.items.add(file);
                            input.files = dt.files;
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                          }
                          return;
                        }
                      }
                    }
                  } catch (err) {
                    console.warn('Clipboard read error:', err);
                  }
                });
              }
            } catch (e) {
              console.debug('Listener setup error:', e);
            }
          }

          setupListeners();
          setTimeout(setupListeners, 400);
          setTimeout(setupListeners, 1000);
        })();
      </script>
    </body>
    </html>
    """, height=82)

# Evaluasi Sumber Citra (Upload vs Clipboard)
active_image = None
active_meta = {}

if uploaded_file is not None:
    file_bytes = uploaded_file.size
    file_mb = file_bytes / (1024 * 1024)

    if file_bytes > MAX_FILE_SIZE_BYTES:
        st.error(f"Ukuran file melebihi batas {int(MAX_FILE_SIZE_MB)} MB (terdeteksi: {file_mb:.2f} MB). Silakan unggah gambar yang lebih kecil.")
    else:
        try:
            active_image = Image.open(io.BytesIO(uploaded_file.read()))
            img_format = active_image.format or uploaded_file.name.split('.')[-1].upper()
            active_meta = {
                "format": img_format,
                "size_mb": file_mb,
            }
        except Exception as err:
            st.error(f"Gagal memproses file citra: {err}")

# Tampilkan Hasil Jika Citra Tersedia
if active_image is not None:
    col_img, col_pred = st.columns([1, 1], gap="medium")

    with col_img:
        try:
            st.image(active_image, width="stretch")
        except TypeError:
            st.image(active_image, use_container_width=True)
        st.markdown(f"""
        <div class="image-meta-bar">
            <span>{active_image.width} × {active_image.height} px</span>
            <span>{active_meta['format']}</span>
            <span>{active_meta['size_mb']:.2f} MB</span>
        </div>
        """, unsafe_allow_html=True)

    with col_pred:
        if model is None:
            st.error(f"File bobot model '{active_cfg['name']}' belum ditemukan. Silakan jalankan proses training di notebook terlebih dahulu.")
        else:
            pred_class, confidence, ai_prob, real_prob = run_inference(
                active_image, model, device, arch=active_cfg.get("arch", "")
            )

            # Logika threshold cerdas & terkalibrasi:
            # - Keyakinan Tinggi (>= 80%)
            # - Keyakinan Sedang (>= 60%)
            # - Daerah Ambigu / Modifikasi Parsial (40% - 59.9%)
            if ai_prob >= 80.0:
                verdict_class = "verdict-ai"
                label_color = "label-ai"
                verdict_title = "AI-Generated Image"
                sub_status = f"Tingkat Keyakinan Tinggi (AI: {ai_prob:.1f}%)"
                badge_status = "AI-Generated"
            elif real_prob >= 80.0:
                verdict_class = "verdict-real"
                label_color = "label-real"
                verdict_title = "Real Image"
                sub_status = f"Tingkat Keyakinan Tinggi (Real: {real_prob:.1f}%)"
                badge_status = "Real Image"
            elif ai_prob >= 60.0:
                verdict_class = "verdict-ai"
                label_color = "label-ai"
                verdict_title = "AI-Generated Image"
                sub_status = f"Tingkat Keyakinan Sedang (AI: {ai_prob:.1f}%)"
                badge_status = "Cenderung AI"
            elif real_prob >= 60.0:
                verdict_class = "verdict-real"
                label_color = "label-real"
                verdict_title = "Real Image"
                sub_status = f"Tingkat Keyakinan Sedang (Real: {real_prob:.1f}%)"
                badge_status = "Cenderung Real"
            else:
                # Keyakinan di rentang ambigu (40% - 59.9%): terdeteksi sebagian diedit/dimodifikasi AI
                verdict_class = "verdict-modified"
                label_color = "label-modified"
                verdict_title = "Gambar ini terdeteksi sebagian diedit atau dimodifikasi menggunakan kecerdasan buatan (AI)"
                sub_status = f"Tingkat Keyakinan Rendah / Ambigu (AI: {ai_prob:.1f}% vs Real: {real_prob:.1f}%)"
                badge_status = "Sebagian Diedit AI"

            st.markdown(f"""
            <div class="verdict-card {verdict_class}">
                <div class="verdict-tag">Hasil Klasifikasi ({active_cfg['name']}) • {sub_status}</div>
                <div class="verdict-label {label_color}">{verdict_title}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-container">
                <div class="stat-card">
                    <div class="stat-title">Tingkat Keyakinan</div>
                    <div class="stat-num">{confidence:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Status Deteksi</div>
                    <div class="stat-num" style="font-size: 1.0rem; padding-top: 0.25rem;">{badge_status}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption("Distribusi Probabilitas")

            st.write(f"AI-Generated: **{ai_prob:.1f}%**")
            st.progress(ai_prob / 100.0)

            st.write(f"Real Image: **{real_prob:.1f}%**")
            st.progress(real_prob / 100.0)
else:
    st.caption("Unggah gambar via drag and drop, file explorer, atau paste langsung dari clipboard.")
