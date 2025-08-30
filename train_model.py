# === File: train_model.py ===
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

print("Memulai proses training model...")

# 1. MEMUAT DATA LENGKAP
print("1. Memuat data dari 'data_lengkap.csv'...")
file_path = os.path.join("Data", "data_lengkap.csv")
df = pd.read_csv(file_path)

# Pastikan hanya baris dengan data yang diperlukan yang digunakan
df.dropna(subset=['cleaned_text', 'sentiment_label'], inplace=True)

X = df['cleaned_text'].astype(str)
le = LabelEncoder()
y = le.fit_transform(df['sentiment_label'])

# 2. VECTORISASI TF-IDF
print("2. Membuat fitur teks dengan TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)

# 3. SMOTE UNTUK MENANGANI DATA TIDAK SEIMBANG
print("3. Menyeimbangkan data dengan SMOTE...")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_tfidf, y)

# 4. MELATIH MODEL LIGHTGBM
print("4. Melatih model LightGBM...")
model = LGBMClassifier(random_state=42)
model.fit(X_bal, y_bal)
print("   -> Model berhasil dilatih!")

# 5. MENYIMPAN MODEL DAN KOMPONEN
print("5. Menyimpan model, vectorizer, dan encoder...")
joblib.dump(model, os.path.join("Data", "lightgbm_model.joblib"))
joblib.dump(tfidf, os.path.join("Data", "tfidf_vectorizer.joblib"))
joblib.dump(le, os.path.join("Data", "label_encoder.joblib"))

print("\nProses selesai! Model dan komponen telah disimpan di dalam folder 'Data'.")