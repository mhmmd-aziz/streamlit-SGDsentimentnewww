# === File: streamlit_app.py ===
import os
# 🔥 Fix untuk Hugging Face Spaces agar bisa nulis config
os.environ["XDG_CONFIG_HOME"] = "/tmp"

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Dashboard Sentimen SDG", page_icon="📈", layout="wide")

@st.cache_resource
def load_model_components():
    """Memuat model, vectorizer, dan encoder dari folder yang sama dengan file ini."""
    try:
        model = joblib.load("lightgbm_model.joblib")
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        encoder = joblib.load("label_encoder.joblib")
        return model, vectorizer, encoder
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_data():
    """Memuat data lengkap dari folder yang sama dengan file ini."""
    try:
        df = pd.read_csv("data_lengkap.csv")
        return df
    except FileNotFoundError:
        return None

model, vectorizer, encoder = load_model_components()
df_ulasan = load_data()

st.title("📈 Dashboard Analisis Sentimen Terkait SDG")
st.write("Analisis sentimen publik terhadap inisiatif brand yang berkaitan dengan *Sustainable Development Goals* (SDG).")

if model is None or df_ulasan is None:
    st.error("❌ Gagal memuat model atau file 'data_lengkap.csv'. Jalankan `lengkapi_data.py` dan `train_model.py` terlebih dahulu.")
    st.stop()

# Mengganti nama kolom 'sentiment_label' menjadi 'sentimen' untuk konsistensi
df_predicted = df_ulasan.rename(columns={"sentiment_label": "sentimen"})
df_predicted.dropna(subset=['sentimen', 'brand', 'sdg_goal'], inplace=True)

st.sidebar.header("Filter Data")
all_brands = df_predicted['brand'].unique().tolist()
selected_brands = st.sidebar.multiselect("Pilih Brand", options=['Semua Brand'] + all_brands, default='Semua Brand')

if 'Semua Brand' in selected_brands or not selected_brands:
    filtered_df = df_predicted
else:
    filtered_df = df_predicted[df_predicted['brand'].isin(selected_brands)]

st.header(f"Insight untuk: {', '.join(selected_brands)}")

if filtered_df.empty:
    st.warning("Tidak ada data untuk brand yang dipilih.")
else:
    st.subheader("📊 Persentase Sentimen per Kategori SDG")
    sentiment_per_sdg = filtered_df.groupby(['sdg_goal', 'sentimen']).size().unstack(fill_value=0)

    for sent_col in ['positif', 'negatif', 'netral']:
        if sent_col not in sentiment_per_sdg.columns:
            sentiment_per_sdg[sent_col] = 0

    sentiment_percentage = sentiment_per_sdg.apply(lambda x: x*100 / sum(x) if sum(x) > 0 else x, axis=1)

    fig_bar = px.bar(
        sentiment_percentage,
        x=sentiment_percentage.index,
        y=['positif', 'negatif', 'netral'],
        title="Distribusi Sentimen pada Setiap SDG",
        labels={'x': 'Kategori SDG', 'value': 'Persentase (%)'},
        color_discrete_map={'positif': 'green', 'negatif': 'red', 'netral': 'grey'},
        barmode='stack'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("🏆 Ranking SDG Paling Positif (Dicintai)")
    if 'positif' in sentiment_percentage.columns:
        ranking_df = sentiment_percentage[['positif']].sort_values(by='positif', ascending=False).reset_index()
        ranking_df['positif'] = ranking_df['positif'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(
            ranking_df,
            column_config={"sdg_goal": "Kategori SDG", "positif": "Persentase Sentimen Positif"},
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("Tidak ada data sentimen 'positif' untuk ditampilkan dalam ranking.")

    with st.expander("Lihat Data Mentah yang Difilter"):
        st.dataframe(filtered_df)
