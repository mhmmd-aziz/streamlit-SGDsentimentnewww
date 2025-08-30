# === File: lengkapi_data.py ===
import pandas as pd
import os
import numpy as np

print("Memulai proses pelabelan data otomatis...")

# --- 1. (WAJIB DIEDIT) DEFINISIKAN KATA KUNCI ANDA DI SINI ---
# Semakin lengkap daftar ini, semakin akurat hasilnya.
BRAND_KEYWORDS = {
    'Unilever': ['unilever', 'plastik', 'sampah', 'daur ulang', 'sachet'],
    'Nestle': ['nestle', 'plastik', 'sampah', 'gula', 'kesehatan'],
    'Danone': ['danone', 'plastik', 'aqua', 'galon', 'lingkungan'],
    'Grab': ['grab', 'emisi', 'driver', 'polusi', 'listrik'],
    'Paragon': ['paragon', 'kosmetik', 'ramah lingkungan', 'animal testing'],
    'Astra': ['astra', 'lingkungan', 'emisi', 'polusi', 'sosial']
}

SDG_KEYWORDS = {
    '12: Responsible Consumption': ['plastik', 'sampah', 'daur ulang', 'sachet', 'lingkungan', 'kosmetik', 'ramah lingkungan', 'animal testing'],
    '3: Good Health and Well-being': ['kesehatan', 'gula'],
    '13: Climate Action': ['emisi', 'polusi', 'iklim', 'karbon'],
    '11: Sustainable Cities and Communities': ['listrik', 'transportasi'],
    '8: Decent Work and Economic Growth': ['driver', 'sosial', 'pekerja'],
    '4: Quality Education': ['pendidikan', 'edukasi', 'sekolah', 'beasiswa'],
    '6: Clean Water and Sanitation': ['air bersih', 'sanitasi', 'aqua']
}

# --- 2. FUNGSI UNTUK MENCARI KATA KUNCI ---
def map_entity(text, keyword_dict, default_value=np.nan):
    """Fungsi untuk mencari kata kunci dalam teks dan mengembalikan nama entitasnya."""
    text_lower = str(text).lower()
    for entity_name, keywords in keyword_dict.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return entity_name
    return default_value

# --- 3. PROSES PEMBACAAN DAN PELABELAN DATA ---
try:
    # Menggunakan file yang Anda unggah: data_with_cleaned_text.csv
    source_file = os.path.join("Data", "data_with_sentiment_labels.csv")
    df = pd.read_csv(source_file)
    print(f"Berhasil memuat {len(df)} baris data dari {source_file}")

    # Menggunakan 'full_text' karena mungkin berisi hashtag atau mention asli.
    text_column = 'full_text' if 'full_text' in df.columns else 'cleaned_text'
    print(f"Menggunakan kolom '{text_column}' untuk analisis kata kunci.")

    print("Mencari kata kunci brand...")
    df['brand'] = df[text_column].apply(lambda text: map_entity(text, BRAND_KEYWORDS))
    
    print("Mencari kata kunci SDG...")
    df['sdg_goal'] = df[text_column].apply(lambda text: map_entity(text, SDG_KEYWORDS))

    # Baris ini dinonaktifkan sementara untuk debugging.
    # Setelah Anda puas dengan kata kunci Anda, hapus tanda '#' di bawah ini.
    # df.dropna(subset=['brand', 'sdg_goal'], inplace=True)
    
    original_rows = len(df)
    labeled_rows = df.dropna(subset=['brand', 'sdg_goal']).shape[0]

    # --- 4. SIMPAN HASIL KE FILE BARU ---
    output_file = os.path.join("Data", "data_lengkap.csv")
    df.to_csv(output_file, index=False)

    print("\n--- Ringkasan ---")
    print(f"Data asli: {original_rows} baris")
    print(f"Baris yang berhasil dilabeli (brand DAN SDG ditemukan): {labeled_rows} baris")
    print(f"Hasil disimpan di file baru: {output_file}")
    print("\nProses selesai! Silakan periksa 'data_lengkap.csv' untuk melihat hasilnya.")

except FileNotFoundError:
    print(f"Error: File '{source_file}' tidak ditemukan. Pastikan file ada di dalam folder 'Data'.")
except Exception as e:
    print(f"Terjadi error: {e}")