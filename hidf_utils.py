import os
import struct
import zlib
import urllib.request
from tqdm import tqdm

ZENODO_BASE = "https://zenodo.org/records/16140829/files"
URL_FAKE_ZIP = f"{ZENODO_BASE}/Fake-img.zip"
URL_REAL_ZIP = f"{ZENODO_BASE}/Real-img.zip"
URL_METADATA = f"{ZENODO_BASE}/metadata.csv"


def stream_extract_zip(url, target_dir, target_count, desc="Ekstraksi Citra"):
    """
    Mengekstrak citra langsung dari HTTP stream ZIP dalam HANYA 1 KONEKSI TUNGGAL.
    Kebal terhadap error HTTP 429 (Too Many Requests) dari Zenodo karena tidak
    membuat ribuan request paralel, melainkan membaca data terkompresi secara sekuensial.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Hitung citra yang sudah ada untuk resume jika terputus
    existing_files = set(f.lower() for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    current_count = len(existing_files)
    
    if current_count >= target_count:
        print(f"[+] Folder '{target_dir}' sudah memiliki {current_count} citra (target: {target_count}). Lewati.")
        return

    needed = target_count - current_count
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=60)
    
    buf = bytearray()
    extracted_now = 0
    
    pbar = tqdm(total=target_count, initial=current_count, desc=desc, unit="img")
    
    try:
        while extracted_now < needed:
            # 1. Pastikan buffer memiliki minimal 30 byte (panjang Local File Header dasar)
            while len(buf) < 30:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                buf.extend(chunk)
                
            if len(buf) < 30:
                break
                
            # Jika signature bukan 'PK\x03\x04', mungkin sudah mencapai direktori sentral atau akhir
            if buf[:4] != b'PK\x03\x04':
                # Cari signature PK\x03\x04 berikutnya jika ada desinkronisasi kecil
                next_sig = buf.find(b'PK\x03\x04', 1)
                if next_sig != -1:
                    del buf[:next_sig]
                    continue
                else:
                    break
                    
            # 2. Parse Local File Header
            ver, flags, method, mtime, mdate, crc, comp_size, uncomp_size, fn_len, ex_len = struct.unpack('<HHHHHIIIHH', buf[4:30])
            total_header = 30 + fn_len + ex_len
            total_entry = total_header + comp_size
            
            # 3. Pastikan seluruh payload file sudah masuk ke buffer
            while len(buf) < total_entry:
                read_size = max(1024 * 512, total_entry - len(buf))
                chunk = resp.read(read_size)
                if not chunk:
                    break
                buf.extend(chunk)
                
            if len(buf) < total_entry:
                break  # Stream terputus sebelum file selesai
                
            fname = buf[30:30+fn_len].decode('utf-8', errors='ignore')
            payload = buf[total_header:total_entry]
            
            # Buang entry yang sudah diproses dari buffer memori agar RAM hemat
            del buf[:total_entry]
            
            # 4. Dekompresi citra JPEG
            if not fname.endswith('/') and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                basename = os.path.basename(fname)
                save_path = os.path.join(target_dir, basename)
                
                # Jika file belum ada, dekompresi dan simpan
                if basename.lower() not in existing_files:
                    try:
                        if method == 8:    # Deflated (ZIP standard)
                            decomp = zlib.decompress(payload, -15)
                        elif method == 0:  # Stored (No compression)
                            decomp = payload
                        else:
                            continue
                            
                        with open(save_path, 'wb') as f:
                            f.write(decomp)
                            
                        existing_files.add(basename.lower())
                        extracted_now += 1
                        pbar.update(1)
                    except Exception:
                        pass
                        
    finally:
        resp.close()
        pbar.close()


def download_hidf_dataset(target_dir="data/hidf", n_samples_per_class=1500, max_workers=None):
    """
    Mengunduh dataset HiDF dengan teknik Single-Connection Stream Extraction.
    Metode ini mencegah HTTP 429 (Too Many Requests) karena hanya membutuhkan
    1 request HTTP untuk seluruh kelas Real, dan 1 request HTTP untuk Fake.
    """
    real_dir = os.path.join(target_dir, "real")
    fake_dir = os.path.join(target_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # 1. Unduh metadata HiDF jika belum ada
    meta_path = os.path.join(target_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        print(f"[*] Mengunduh metadata HiDF dari {URL_METADATA}...")
        try:
            urllib.request.urlretrieve(URL_METADATA, meta_path)
            print(f"[+] Metadata tersimpan di {meta_path}")
        except Exception as e:
            print(f"[!] Gagal mengunduh metadata: {e}")

    # 2. Cek status lokal
    existing_real = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    existing_fake = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"[*] Status Dataset HiDF di '{target_dir}':")
    print(f"    - Citra Real (Asli) terunduh    : {len(existing_real)} / {n_samples_per_class}")
    print(f"    - Citra Fake (Deepfake) terunduh: {len(existing_fake)} / {n_samples_per_class}")

    if len(existing_fake) >= n_samples_per_class and len(existing_real) >= n_samples_per_class:
        print("[+] Dataset HiDF lokal sudah lengkap dan siap digunakan!")
        return real_dir, fake_dir

    # 3. Stream Ekstraksi Citra Fake (Deepfake) - 1 KONEKSI TUNGGAL
    if len(existing_fake) < n_samples_per_class:
        print(f"\n[*] Mengunduh {n_samples_per_class} citra Fake (Deepfake) via Single Stream...")
        stream_extract_zip(
            url=URL_FAKE_ZIP,
            target_dir=fake_dir,
            target_count=n_samples_per_class,
            desc="Mengunduh Fake (Deepfake)"
        )

    # 4. Stream Ekstraksi Citra Real (Asli) - 1 KONEKSI TUNGGAL
    if len(existing_real) < n_samples_per_class:
        print(f"\n[*] Mengunduh {n_samples_per_class} citra Real (Asli) via Single Stream...")
        stream_extract_zip(
            url=URL_REAL_ZIP,
            target_dir=real_dir,
            target_count=n_samples_per_class,
            desc="Mengunduh Real (Asli)"
        )

    total_real = len([f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_fake = len([f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\n[+] Sukses! Dataset HiDF siap di '{target_dir}'.")
    print(f"    Total: {total_real} Real + {total_fake} Fake = {total_real + total_fake} citra.")
    return real_dir, fake_dir


if __name__ == "__main__":
    download_hidf_dataset(target_dir="data/hidf", n_samples_per_class=10)
