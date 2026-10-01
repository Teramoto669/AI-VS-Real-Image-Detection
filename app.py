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

    .verdict-tag {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 0.35rem;
    }

    .verdict-label {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
    }

    .label-ai {
        color: #FB7185;
    }

    .label-real {
        color: #34D399;
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

    /* Sidebar spec rows */
    .spec-row {
        display: flex;
        justify-content: space-between;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.85rem;
    }

    .spec-key {
        color: #64748B;
    }

    .spec-val {
        color: #E2E8F0;
        font-weight: 500;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript Listener untuk interaksi Drag-and-Drop & Paste (Ctrl+V) langsung pada dokumen
components.html("""
<script>
(function() {
    function setupListeners() {
        try {
            const parentDoc = window.parent.document;
            if (!parentDoc || parentDoc._imageListenersAttached) return;
            parentDoc._imageListenersAttached = true;

            function getFileInput() {
                return parentDoc.querySelector('section[data-testid="stFileUploaderDropzone"] input[type="file"]') ||
                       parentDoc.querySelector('input[type="file"]');
            }

            // Global Ctrl+V (Paste) handler
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

            // Global Dragover & Drop handler agar browser tidak membuka file langsung
            parentDoc.addEventListener('dragover', function(e) {
                e.preventDefault();
            });

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
        } catch (e) {
            console.debug('Listener setup:', e);
        }
    }

    setupListeners();
    setTimeout(setupListeners, 400);
    setTimeout(setupListeners, 1200);
})();
</script>
""", height=0, width=0)

MODEL_PATH = "best_efficientnet_b0_ai_vs_real.pth"
MAX_FILE_SIZE_MB = 2.0
MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)

# Pipeline Transformasi
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource(show_spinner=False)
def get_model():
    """Memuat model EfficientNet-B0 dan checkpoint bobot."""
    if not os.path.exists(MODEL_PATH):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2)
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device


def run_inference(image: Image.Image, model: nn.Module, device: torch.device):
    """Menjalankan inferensi dan menghitung probabilitas tiap kelas."""
    rgb_image = image.convert("RGB")
    tensor_img = transform_pipeline(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probs, 0)

    ai_prob = probs[0].item() * 100.0
    real_prob = probs[1].item() * 100.0

    return predicted_idx.item(), confidence.item() * 100.0, ai_prob, real_prob


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### Spesifikasi")

    model, device = get_model()
    device_label = "CUDA" if (device and device.type == "cuda") else "CPU"

    st.markdown(f"""
    <div style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
        <div class="spec-row"><span class="spec-key">Status</span><span class="spec-val">{'Siap' if model else 'Offline'}</span></div>
        <div class="spec-row"><span class="spec-key">Model</span><span class="spec-val">EfficientNet-B0</span></div>
        <div class="spec-row"><span class="spec-key">Perangkat</span><span class="spec-val">{device_label}</span></div>
        <div class="spec-row"><span class="spec-key">Input Res</span><span class="spec-val">224 × 224 px</span></div>
        <div class="spec-row"><span class="spec-key">Batas File</span><span class="spec-val">Maks. 2 MB</span></div>
        <div class="spec-row"><span class="spec-key">Akurasi Val</span><span class="spec-val">97.3%</span></div>
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
        help="Batas ukuran maksimal 2 MB (format JPG, PNG, WEBP)"
    )

with col_paste:
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
          <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
          <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
          <path d="M9 12h6"></path>
          <path d="M9 16h6"></path>
        </svg>
      </button>
      <script>
        document.getElementById('pasteActionBtn').addEventListener('click', async function() {
          try {
            const parentDoc = window.parent.document;
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
                  const input = parentDoc.querySelector('section[data-testid="stFileUploaderDropzone"] input[type="file"]') ||
                                parentDoc.querySelector('input[type="file"]');
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
        st.error(f"Ukuran file melebihi batas 2 MB (terdeteksi: {file_mb:.2f} MB). Silakan unggah gambar yang lebih kecil.")
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
            st.error("Model checkpoint tidak ditemukan.")
        else:
            pred_class, confidence, ai_prob, real_prob = run_inference(active_image, model, device)
            is_ai = (pred_class == 0)

            verdict_class = "verdict-ai" if is_ai else "verdict-real"
            label_color = "label-ai" if is_ai else "label-real"
            label_text = "AI-Generated" if is_ai else "Real Image"

            st.markdown(f"""
            <div class="verdict-card {verdict_class}">
                <div class="verdict-tag">Hasil Klasifikasi</div>
                <div class="verdict-label {label_color}">{label_text}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-container">
                <div class="stat-card">
                    <div class="stat-title">Keyakinan Prediksi</div>
                    <div class="stat-num">{confidence:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Kelas Terpilih</div>
                    <div class="stat-num" style="font-size: 1.05rem; padding-top: 0.25rem;">{label_text}</div>
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
