import ast
import os
import pickle
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

st.set_page_config(
    page_title="Skincare Recommendation System",
    page_icon="🧴",
    layout="wide",
)

# Tema Premium Khas Beauty App (Soft Pink, Rose Gold, Deep Maroon)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #fffafd 0%, #fdf4f8 50%, #ffffff 100%);
}

.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px;
}

/* ====== SIDEBAR STYLING ====== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffeef4 0%, #ffdbe7 50%, #ffffff 100%) !important;
    border-right: 1px solid #fbcfe8 !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem !important;
}

.sidebar-section-title {
    font-size: 11px;
    font-weight: 800;
    color: #9d174d;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 20px 0 10px 8px;
}

section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100%;
    padding: 12px 16px !important;
    border-radius: 14px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(249, 168, 212, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #ffffff;
    border: 1px solid #f472b6;
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(219, 39, 119, 0.08);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-weight: 700 !important;
    color: #831843 !important;
    font-size: 14px !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #ec4899 0%, #be185d 100%) !important;
    border: none !important;
    box-shadow: 0 8px 20px rgba(190, 24, 93, 0.2);
    transform: translateX(5px);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* ====== CUSTOM UI COMPONENTS ====== */
.hero-banner {
    background: linear-gradient(135deg, #be185d 0%, #9d174d 50%, #630b2c 100%);
    padding: 35px 40px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 12px 30px rgba(157, 23, 77, 0.25);
    margin-bottom: 30px;
}
.hero-title {
    font-size: 28px;
    font-weight: 800;
    line-height: 1.3;
    margin: 0 0 10px 0;
}
.hero-subtitle {
    font-size: 15px;
    opacity: 0.9;
    font-weight: 400;
    line-height: 1.6;
    margin: 0;
}

.custom-card {
    background: #ffffff;
    border: 1px solid #fbcfe8;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 4px 18px rgba(190, 24, 93, 0.04);
    height: 100%;
    transition: transform 0.3s ease;
}
.custom-card:hover {
    transform: translateY(-3px);
}
.card-label {
    font-size: 12px;
    color: #851c44;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.card-value {
    font-size: 22px;
    color: #be185d;
    font-weight: 800;
    word-wrap: break-word;
    line-height: 1.3;
}

.product-card {
    background: #ffffff;
    border: 1px solid #f3e8ff;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
    border-left: 5px solid #ec4899;
}
.product-title {
    color: #9d174d;
    font-size: 18px;
    font-weight: 800;
    margin: 0 0 8px 0;
}
.tag-container {
    margin-top: 12px;
}
.tag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 8px;
    background: #fff1f7;
    color: #db2777;
    margin-right: 6px;
    margin-bottom: 6px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #fbcfe8;
}

.stButton > button {
    background: linear-gradient(135deg, #ec4899 0%, #be185d 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(236, 72, 153, 0.25) !important;
}
</style>
""", unsafe_allow_html=True)

DEFAULT_CLUSTER_LABELS = {
    0: "Exfoliating & Brightening",
    1: "Hydrating & Anti-Aging",
    2: "Acne & Purifying",
}

DEFAULT_SKIN_CONCERN_MAPPING = {
    ("Oily", "Acne"): "Acne & Purifying",
    ("Oily", "Brightening"): "Exfoliating & Brightening",
    ("Oily", "Anti-Aging"): "Hydrating & Anti-Aging",
    ("Oily", "Hydration"): "Hydrating & Anti-Aging",
    ("Dry", "Acne"): "Acne & Purifying",
    ("Dry", "Brightening"): "Exfoliating & Brightening",
    ("Dry", "Anti-Aging"): "Hydrating & Anti-Aging",
    ("Dry", "Hydration"): "Hydrating & Anti-Aging",
    ("Combination", "Acne"): "Acne & Purifying",
    ("Combination", "Brightening"): "Exfoliating & Brightening",
    ("Combination", "Anti-Aging"): "Hydrating & Anti-Aging",
    ("Combination", "Hydration"): "Hydrating & Anti-Aging",
    ("Sensitive", "Acne"): "Acne & Purifying",
    ("Sensitive", "Brightening"): "Exfoliating & Brightening",
    ("Sensitive", "Anti-Aging"): "Hydrating & Anti-Aging",
    ("Sensitive", "Hydration"): "Hydrating & Anti-Aging",
}

FACE_SKINCARE_TYPES = ["serum", "cleanser", "moisturiser", "toner", "mask", "eye cream", "essence", "sunscreen"]

# PATH MANAJEMEN
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent
BASE_DIR = APP_DIR.parent

DATA_CANDIDATES = [
    BASE_DIR / "dataset" / "skincare_clustered.csv",
    BASE_DIR / "skincare_clustered.csv",
    APP_DIR / "dataset" / "skincare_clustered.csv",
    APP_DIR / "skincare_clustered.csv",
]
MODEL_DIR_CANDIDATES = [BASE_DIR / "model", APP_DIR / "model"]
ASSET_DIR_CANDIDATES = [BASE_DIR / "assets", APP_DIR / "assets"]

# ======================================================
# UTILITIES
# ======================================================
def find_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None

def find_model_path(filename):
    for model_dir in MODEL_DIR_CANDIDATES:
        path = model_dir / filename
        if path.exists():
            return path
    return None

def find_asset_path(filename):
    for asset_dir in ASSET_DIR_CANDIDATES:
        path = asset_dir / filename
        if path.exists():
            return path
    return None

def load_pickle_or_joblib(filename, loader="pickle"):
    path = find_model_path(filename)
    if path is None:
        return None
    try:
        if loader == "joblib":
            return joblib.load(path)
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def clean_ingredient_name(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ", "_")
    return text

def parse_ingredients(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    value = str(value).strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [item.strip() for item in value.split(",") if item.strip()]

def ingredients_list_to_str(ingredients_list):
    return " ".join([clean_ingredient_name(item) for item in ingredients_list])

def infer_price_category(price_idr):
    try:
        price_idr = float(price_idr)
    except Exception:
        return np.nan
    if price_idr < 150000:
        return "Low"
    if price_idr <= 500000:
        return "Medium"
    return "High"

def format_rupiah(value):
    try:
        return f"Rp {float(value):,.0f}".replace(",", ".")
    except Exception:
        return "-"

@st.cache_data(show_spinner=False)
def load_dataset_cached():
    data_path = find_existing_path(DATA_CANDIDATES)
    if data_path is None:
        return pd.DataFrame(), None
    df = pd.read_csv(data_path)
    
    if "ingredients_list" not in df.columns:
        if "clean_ingreds" in df.columns:
            df["ingredients_list"] = df["clean_ingreds"].apply(parse_ingredients)
        elif "ingredients_str" in df.columns:
            df["ingredients_str"] = df["ingredients_str"].astype(str)
            df["ingredients_list"] = df["ingredients_str"].str.replace("_", " ").str.split()
        else:
            df["ingredients_list"] = [[] for _ in range(len(df))]
    else:
        df["ingredients_list"] = df["ingredients_list"].apply(parse_ingredients)

    if "ingredients_str" not in df.columns:
        df["ingredients_str"] = df["ingredients_list"].apply(ingredients_list_to_str)
    if "num_ingredients" not in df.columns:
        df["num_ingredients"] = df["ingredients_list"].apply(len)
        
    df["num_ingredients"] = pd.to_numeric(df["num_ingredients"], errors="coerce").fillna(0).astype(int)
    df["price_idr"] = pd.to_numeric(df.get("price_idr", np.nan), errors="coerce")
    
    if "price_category" not in df.columns:
        df["price_category"] = df["price_idr"].apply(infer_price_category)
    if "product_type" not in df.columns:
        df["product_type"] = "Unknown"
    if "product_name" not in df.columns:
        df["product_name"] = "Produk Tanpa Nama"
    if "cluster_label" not in df.columns:
        df["cluster_label"] = df.get("cluster", pd.Series([])).map(DEFAULT_CLUSTER_LABELS).fillna("Belum diketahui")
        
    df["is_face_skincare"] = df["product_type"].str.lower().isin(FACE_SKINCARE_TYPES)
    return df, str(data_path)

@st.cache_resource(show_spinner=False)
def load_models_cached():
    cluster_labels = load_pickle_or_joblib("cluster_labels.pkl", loader="pickle") or DEFAULT_CLUSTER_LABELS
    skin_concern_mapping = load_pickle_or_joblib("skin_concern_mapping.pkl", loader="pickle") or DEFAULT_SKIN_CONCERN_MAPPING
    kmeans_model = load_pickle_or_joblib("kmeans_model.pkl", loader="pickle")
    tfidf_vectorizer = load_pickle_or_joblib("tfidf_vectorizer.pkl", loader="pickle")
    classification_model = load_pickle_or_joblib("classification_best_model_dataset_terbaru.pkl", loader="joblib")

    return {
        "cluster_labels": cluster_labels,
        "skin_concern_mapping": skin_concern_mapping,
        "kmeans_model": kmeans_model,
        "tfidf_vectorizer": tfidf_vectorizer,
        "classification_model": classification_model,
    }

def init_session_data():
    df, data_path = load_dataset_cached()
    if "data" not in st.session_state:
        st.session_state["data"] = df.copy()
    if "data_path" not in st.session_state:
        st.session_state["data_path"] = data_path

def show_product_card(row):
    ingredients = parse_ingredients(row.get("ingredients_list", []))
    tags = "".join([f"<span class='tag'>{item}</span>" for item in ingredients[:8]])
    st.markdown(f"""
    <div class='product-card'>
        <h4 class='product-title'>✨ {row.get('product_name', '-')}</h4>
        <div style='font-size: 13px; color: #4b5563; line-height: 1.6;'>
            <b>Jenis Produk:</b> {row.get('product_type', '-').title()} | 
            <b>Cluster Label:</b> <span style='color:#be185d; font-weight:700;'>{row.get('cluster_label', '-')}</span> <br>
            <b>Harga:</b> <span style='color:#10b981; font-weight:700;'>{format_rupiah(row.get('price_idr', np.nan))}</span> ({row.get('price_category', '-')} Price)
        </div>
        <div class='tag-container'>
            {tags if tags else '<span class="tag">Tidak ada kandungan bahan</span>'}
        </div>
    </div>
    """, unsafe_allow_html=True)

def make_new_product_row(product_name, product_type, ingredients_text, price_idr, predicted_label):
    ingredients_list = [item.strip() for item in ingredients_text.split(",") if item.strip()]
    ingredients_str = ingredients_list_to_str(ingredients_list)
    return {
        "product_name": product_name,
        "product_type": product_type,
        "clean_ingreds": str(ingredients_list),
        "ingredients_list": ingredients_list,
        "num_ingredients": len(ingredients_list),
        "ingredients_str": ingredients_str,
        "price_idr": float(price_idr),
        "price_category": infer_price_category(price_idr),
        "cluster_label": predicted_label,
        "is_face_skincare": str(product_type).lower() in FACE_SKINCARE_TYPES,
    }

def predict_new_product(models, product_type, ingredients_text, price_idr):
    ingredients_list = [item.strip() for item in ingredients_text.split(",") if item.strip()]
    ingredients_str = ingredients_list_to_str(ingredients_list)
    price_cat = infer_price_category(price_idr)
    
    input_df = pd.DataFrame([{
        "ingredients_str": ingredients_str,
        "num_ingredients": len(ingredients_list),
        "price_idr": float(price_idr),
        "price_category": price_cat,
        "product_type": product_type,
    }])

    class_prediction = None
    cluster_prediction = None

    if models["classification_model"] is not None:
        try:
            class_prediction = models["classification_model"].predict(input_df)[0]
        except Exception:
            class_prediction = None

    if models["kmeans_model"] is not None and models["tfidf_vectorizer"] is not None:
        try:
            X_new = models["tfidf_vectorizer"].transform([ingredients_str])
            cluster_id = int(models["kmeans_model"].predict(X_new)[0])
            cluster_prediction = models["cluster_labels"].get(cluster_id, f"Cluster {cluster_id}")
        except Exception:
            cluster_prediction = None

    final_prediction = class_prediction or cluster_prediction or "Belum dapat diprediksi"
    return final_prediction, class_prediction, cluster_prediction, input_df

def save_dataset_to_csv(df):
    data_path = st.session_state.get("data_path") or str(BASE_DIR / "dataset" / "skincare_clustered.csv")
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    return data_path

def draw_bar_chart(data, x, y, title, color_scale=None):
    if px is not None:
        if color_scale is None:
            color_scale = px.colors.sequential.RdPu
        fig = px.bar(data, x=x, y=y, title=title, text=y, color=y, color_continuous_scale=color_scale)
        fig.update_layout(xaxis_title="", yaxis_title="Jumlah", title_x=0.01, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(data.set_index(x)[y])

def page_home(df, models):
    st.markdown(f"""
    <div class="hero-banner">
        <h1 class="hero-title">🧴 Sistem Rekomendasi Produk Skincare Menggunakan Clustering dan Classification</h1>
        <p class="hero-subtitle">Aplikasi web sederhana untuk merekomendasikan produk berdasarkan tipe kulit, permasalahan kulit, ingredients, jenis produk, dan kategori harga.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        jumlah_produk = f"{len(df):,}" if not df.empty else "1.139"
        st.markdown(f'<div class="custom-card"><div class="card-label">Jumlah Produk</div><div class="card-value">{jumlah_produk}</div></div>', unsafe_allow_html=True)
    with c2:
        varian_jenis = df["product_type"].nunique() if not df.empty else "15"
        st.markdown(f'<div class="custom-card"><div class="card-label">Jumlah Jenis Produk</div><div class="card-value">{varian_jenis}</div></div>', unsafe_allow_html=True)
    with c3:
        total_klaster = df["cluster_label"].nunique() if not df.empty else "3"
        st.markdown(f'<div class="custom-card"><div class="card-label">Jumlah Cluster</div><div class="card-value">{total_klaster}</div></div>', unsafe_allow_html=True)
    with c4:
        status_model = "Aktif" if models["classification_model"] is not None else "Non-Aktif"
        st.markdown(f'<div class="custom-card"><div class="card-label">Model Classification</div><div class="card-value">{status_model}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 💡 Deskripsi Proyek")
        st.write(
            "Proyek ini menerapkan dua metode Data Mining, yaitu K-Means Clustering untuk mengelompokkan "
            "produk berdasarkan komposisi ingredients dan Classification untuk memprediksi kelompok rekomendasi "
            "produk baru. Hasil model kemudian diimplementasikan dalam aplikasi web berbasis Streamlit."
        )
        
        st.markdown("### 👥 Identitas Anggota")
        st.markdown("- **Zakia Ramadhani (24051214081)**")
        st.markdown("- **Komang Nency Astiti (24051214082)**")
            
    with col_b:
        st.markdown("### 🚀 Fitur Utama Aplikasi")
        
        tab_cust, tab_adm, tab_vis, tab_abt = st.tabs(["👤 Customer", "🛠️ Admin", "📈 Visualisasi", "ℹ️ Tentang"])
        with tab_cust:
            st.markdown("""
            * Input tipe kulit dan permasalahan kulit.
            * Memilih cakupan rekomendasi: Face atau Face and Body.
            * Memilih kategori harga dan jumlah rekomendasi.
            * Melihat daftar produk yang paling sesuai.
            """)
        with tab_adm:
            st.markdown("""
            * Menambahkan produk skincare/bodycare baru.
            * Sistem memprediksi cluster produk.
            * Produk baru dapat disimpan ke database aktif.
            """)
        with tab_vis:
            st.markdown("""
            * Grafik pendukung dataset.
            * Visualisasi hasil analisis clustering dan classification.
            """)
        with tab_abt:
            st.markdown("""
            * Penjelasan metode Data Mining.
            * Kumpulan data yang digunakan.
            * Informasi proyek dan alur sistem.
            """)

def page_dataset(df):
    st.markdown('<h2 style="margin-top:0;">📊 Gambaran Umum Dataset</h2>', unsafe_allow_html=True)
    if df.empty:
        st.error("Dataset aktif kosong atau gagal dimuat.")
        return

    # Hitung nilai asli secara dinamis agar presisi pas demo UAS
    val_rows = df.shape[0]
    val_cols = df.shape[1]
    val_missing = int(df.isna().sum().sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="custom-card"><div class="card-label">Jumlah Baris</div><div class="card-value">{val_rows:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="custom-card"><div class="card-label">Jumlah Kolom</div><div class="card-value">{val_cols}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="custom-card"><div class="card-label">Missing Value</div><div class="card-value">{val_missing}</div></div>', unsafe_allow_html=True)

    st.write("")
    
    st.markdown("### 🔍 Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 🔢 Statistik Numerik")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("### 🔠 Distribusi Kolom Kategorikal")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown("**Jenis Produk**")
        if "product_type" in df.columns:
            st.dataframe(df["product_type"].value_counts().reset_index(), use_container_width=True)
    with col_k2:
        st.markdown("**Cluster Label**")
        if "cluster_label" in df.columns:
            st.dataframe(df["cluster_label"].value_counts().reset_index(), use_container_width=True)

def page_customer_recommendation(df, models):
    st.markdown('<h2 style="margin-top:0;">🎯 Prediksi / Rekomendasi Customer</h2>', unsafe_allow_html=True)
    if df.empty:
        st.error("Fitur mati karena kegagalan pemuatan dataset internal.")
        return

    mapping = models["skin_concern_mapping"]

    with st.form("form_rekomendasi_customer"):
        col1, col2 = st.columns(2)
        with col1:
            skin_type = st.selectbox("Tipe Karakteristik Kulit Anda:", ["Oily", "Dry", "Combination", "Sensitive"])
            concern = st.selectbox("Target Masalah / Keluhan Utama:", ["Acne", "Brightening", "Anti-Aging", "Hydration"])
            product_types = ["Semua"] + sorted(df["product_type"].dropna().astype(str).unique().tolist())
            product_type = st.selectbox("Sediaan Kategori Produk:", product_types)
        with col2:
            price_categories = ["Semua"] + sorted(df["price_category"].dropna().astype(str).unique().tolist())
            price_category = st.selectbox("Klasifikasi Range Harga:", price_categories)
            product_scope = st.selectbox("Cakupan Area Pemakaian:", ["Face (Skincare)", "Face and Body (Skincare + Bodycare)"])
            top_n = st.slider("Kuantitas Batas Output Produk:", min_value=3, max_value=20, value=10)

        sort_by = st.selectbox("Metode Pengurutan Tampilan:", ["Harga termurah", "Harga termahal", "Jumlah ingredients terbanyak", "Acak"])
        submitted = st.form_submit_button("🔍 Jalankan Komparasi Algoritma Rekomendasi")

    if submitted:
        target_label = mapping.get((skin_type, concern), "Hydrating & Anti-Aging")
        result = df[df["cluster_label"] == target_label].copy()

        if product_type != "Semua":
            result = result[result["product_type"].astype(str) == product_type]
        if price_category != "Semua":
            result = result[result["price_category"].astype(str) == price_category]
        if product_scope == "Face (Skincare)" and "is_face_skincare" in result.columns:
            result = result[result["is_face_skincare"] == True]

        if sort_by == "Harga termurah":
            result = result.sort_values("price_idr", ascending=True)
        elif sort_by == "Harga termahal":
            result = result.sort_values("price_idr", ascending=False)
        elif sort_by == "Jumlah ingredients terbanyak":
            result = result.sort_values("num_ingredients", ascending=False)
        else:
            result = result.sample(frac=1, random_state=42) if len(result) > 0 else result

        result = result.head(top_n)
        st.success(f"🎯 Kombinasi Kulit '{skin_type}' + Concern '{concern}' berpasangan dengan Target Klaster: **{target_label}**.")
        
        for _, row in result.iterrows():
            show_product_card(row)

def page_admin_input(df, models):
    st.markdown('<h2 style="margin-top:0;">🛠️ Admin: Input Produk Baru</h2>', unsafe_allow_html=True)

    existing_types = sorted(df["product_type"].dropna().astype(str).unique().tolist()) if not df.empty and "product_type" in df.columns else []
    default_types = sorted(set(existing_types + ["serum", "cleanser", "moisturiser", "toner", "mask", "eye cream", "essence", "body wash", "sunscreen"]))

    with st.form("form_input_produk"):
        product_name = st.text_input("Nama Varian Produk:", placeholder="Contoh: Skintific Panthenol Soothing Gel")
        product_type = st.selectbox("Jenis Sediaan Produk:", default_types if default_types else ["serum", "cleanser"])
        ingredients_text = st.text_area("Formulasi Kandungan Kimia (Pisahkan dengan tanda koma):", placeholder="Contoh: panthenol, centella asiatica, ceramide, glycerin", height=120)
        price_idr = st.number_input("Harga Jual Nominal Rupiah (IDR):", min_value=0, value=150000, step=5000)
        submitted = st.form_submit_button("⚡ Hitung Prediksi Vektor Model")

    if submitted:
        if not product_name.strip() or not ingredients_text.strip():
            st.error("Gagal! Atribut Nama Produk dan Kandungan Komposisi Bahan wajib diisi lengkap.")
            return

        final_prediction, class_prediction, cluster_prediction, input_df = predict_new_product(
            models=models, product_type=product_type, ingredients_text=ingredients_text, price_idr=price_idr
        )

        st.session_state["last_prediction"] = {
            "final": final_prediction,
            "class": class_prediction,
            "cluster": cluster_prediction,
            "df": input_df
        }
        st.session_state["new_product_row"] = make_new_product_row(product_name, product_type, ingredients_text, price_idr, final_prediction)

    if "last_prediction" in st.session_state:
        pred_data = st.session_state["last_prediction"]
        st.write("")
        st.subheader("Hasil Olah Prediksi Vektor")

        st.markdown("""
        <style>
        .res-card { background-color: #fff3f7; border: 1px solid #fbcfe8; border-radius: 14px; padding: 18px; text-align: center; height: 100%; box-shadow: 0 4px 12px rgba(190,24,93,0.03); }
        .res-lbl { font-size: 11px; color: #6b7280; text-transform: uppercase; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px; }
        .res-val { font-size: 16px; color: #db2777; font-weight: 800; word-wrap: break-word; line-height: 1.4; }
        </style>
        """, unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f'<div class="res-card"><div class="res-lbl">Keputusan Konsensus</div><div class="res-val">{pred_data["final"]}</div></div>', unsafe_allow_html=True)
        with rc2:
            c_val = pred_data["class"] if pred_data["class"] is not None else "Tidak tersedia"
            st.markdown(f'<div class="res-card"><div class="res-lbl">Klasifikasi (Supervised)</div><div class="res-val">{c_val}</div></div>', unsafe_allow_html=True)
        with rc3:
            cl_val = pred_data["cluster"] if pred_data["cluster"] is not None else "Tidak tersedia"
            st.markdown(f'<div class="res-card"><div class="res-lbl">Klasterisasi (K-Means)</div><div class="res-val">{cl_val}</div></div>', unsafe_allow_html=True)

        st.write("")
        st.write("**Matriks Struktur DataFrame Input Model:**")
        st.dataframe(pred_data["df"], use_container_width=True)

    if "new_product_row" in st.session_state:
        st.divider()
        st.subheader("🔒 Komit Data ke Server")
        if st.button("💾 Simpan Permanen ke Dataset Aktif"):
            new_row = st.session_state["new_product_row"]
            combined = pd.concat([st.session_state["data"], pd.DataFrame([new_row])], ignore_index=True)
            st.session_state["data"] = combined
            out_path = save_dataset_to_csv(combined)
            st.success(f"Sukses! Produk berhasil ditambahkan secara fisik ke direktori `{out_path}`.")
            
            del st.session_state["new_product_row"]
            if "last_prediction" in st.session_state:
                del st.session_state["last_prediction"]
            st.rerun()

def page_visualization(df):
    st.markdown('<h2 style="margin-top:0;">📈 Visualisasi</h2>', unsafe_allow_html=True)
    st.write("Halaman ini berisi grafik pendukung dan visualisasi hasil analisis untuk membantu membaca pola dataset, distribusi harga, jenis produk, serta hasil clustering.")
    
    st.divider()
    
    st.subheader("📊 Grafik Pendukung")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Distribusi Kategori Harga**")
        p_dist = df["price_category"].value_counts().reset_index()
        p_dist.columns = ["price_category", "jumlah"]
        draw_bar_chart(p_dist, "price_category", "jumlah", "")
        
    with col_g2:
        st.markdown("**Distribusi Jenis Produk**")
        pt_dist = df["product_type"].value_counts().head(15).reset_index()
        pt_dist.columns = ["product_type", "jumlah"]
        color_sunset = px.colors.sequential.Sunset if px is not None else None
        draw_bar_chart(pt_dist, "product_type", "jumlah", "", color_sunset)

    st.write("")
    st.markdown("**Distribusi Cluster**")
    if "cluster_label" in df.columns:
        cl_dist = df["cluster_label"].value_counts().reset_index()
        cl_dist.columns = ["cluster_label", "jumlah"]
        draw_bar_chart(cl_dist, "cluster_label", "jumlah", "")

    st.write("")
    st.divider()
    st.subheader("🔬 Visualisasi Hasil Analisis")
    
    images_to_load = [
        ("price_category_distribution.png", "Distribusi Kategori Harga"),
        ("product_type_distribution.png", "Distribusi Jenis Produk"),
        ("elbow_silhouette.png", "Metode Elbow & Silhouette"),
        ("cluster_pca.png", "Visualisasi Klaster PCA 2D"),
        ("cluster_actives.png", "Bahan Aktif Terbanyak per Cluster"),
        ("cluster_product_type.png", "Proporsi Jenis Produk per Cluster")
    ]
    
    for filename, caption in images_to_load:
        img_path = find_asset_path(filename)
        if img_path:
            st.image(str(img_path), caption=f"{caption} ({filename})", use_container_width=True)
            st.write("")

def page_about(models):
    st.markdown('<h2 style="margin-top:0;">ℹ️ Tentang Proyek</h2>', unsafe_allow_html=True)
    
    st.markdown("### 📌 Informasi Proyek")
    st.write(
        "Proyek ini membangun sistem rekomendasi produk skincare dan bodycare berbasis Data Mining. "
        "Aplikasi memiliki dua pengguna utama, yaitu customer untuk mendapatkan rekomendasi produk dan admin untuk menambahkan produk baru ke database."
    )
    st.write(
        "Framework kerja yang digunakan mengikuti alur CRISP-DM, mulai dari pemahaman masalah, pemahaman data, "
        "preprocessing, pemodelan, evaluasi, sampai implementasi model dalam aplikasi web."
    )
    
    st.markdown("### 📁 Kumpulan Data")
    st.write(
        "Dataset utama adalah `skincare_clustered.csv`, yaitu dataset hasil preprocessing dan clustering dari notebook. "
        "Dataset berisi informasi produk seperti nama produk, jenis produk, daftar ingredients, jumlah ingredients, harga, kategori harga, dan label cluster."
    )
    st.write(
        "Produk baru yang dimasukkan admin akan disimpan kembali ke dataset aktif sehingga dapat digunakan sebagai database rekomendasi pada proses berikutnya."
    )
    
    st.markdown("### 🔬 Penjelasan Metode")
    st.write(
        "**Clustering** dilakukan menggunakan algoritma K-Means dengan fitur TF-IDF dari ingredients. "
        "Tujuannya adalah mengelompokkan produk berdasarkan kemiripan komposisi bahan. Label cluster yang digunakan adalah "
        "Exfoliating & Brightening, Hydrating & Anti-Aging, serta Acne & Purifying."
    )
    st.write(
        "**Classification** dilakukan menggunakan pipeline preprocessing dan model terbaik dari perbandingan Decision Tree dan Random Forest. "
        "Fitur yang digunakan adalah ingredients_str, num_ingredients, price_idr, price_category, dan product_type, sedangkan targetnya adalah cluster_label."
    )
    
    st.markdown("### 🔄 Alur Sistem")
    st.markdown("""
    1. Dataset skincare dibersihkan dan harga dikonversi ke Rupiah.
    2. Produk dikelompokkan menggunakan K-Means berdasarkan ingredients.
    3. Label cluster digunakan sebagai target model classification.
    4. Customer memasukkan tipe kulit dan concern. Sistem menentukan cluster rekomendasi.
    5. Admin dapat memasukkan produk baru. Sistem memprediksi cluster produk tersebut.
    """)
    
    st.markdown("### 🔒 Status File Model")
    status = pd.DataFrame([
        {"Komponen File": "classification_best_model_dataset_terbaru.pkl", "Status Integrasi": "Terbaca (Sukses)" if models["classification_model"] is not None else "Gagal / Kosong"},
        {"Komponen File": "kmeans_model.pkl", "Status Integrasi": "Terbaca (Sukses)" if models["kmeans_model"] is not None else "Gagal / Kosong"},
        {"Komponen File": "tfidf_vectorizer.pkl", "Status Integrasi": "Terbaca (Sukses)" if models["tfidf_vectorizer"] is not None else "Gagal / Kosong"},
    ])
    st.dataframe(status, use_container_width=True)

def main():
    init_session_data()
    df = st.session_state["data"]
    models = load_models_cached()

    st.sidebar.markdown("""
    <div style='padding: 16px; margin-bottom: 18px; border-radius: 18px; background: linear-gradient(135deg, #ec4899 0%, #be185d 100%); color: white; box-shadow: 0 8px 20px rgba(190, 24, 93, 0.15);'>
        <div style='font-size: 18px; font-weight: 800;'>GlowSkincare</div>
        <div style='font-size: 11px; opacity: 0.8;'>RData Mining Recomendation</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="sidebar-section-title">Navigasi Utama</div>', unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Pilih Halaman",
        ["Beranda", "Gambaran Umum Dataset", "Prediksi / Rekomendasi Customer", "Admin Input Produk", "Visualisasi", "Tentang"],
        label_visibility="collapsed"
    )

    if page == "Beranda":
        page_home(df, models)
    elif page == "Gambaran Umum Dataset":
        page_dataset(df)
    elif page == "Prediksi / Rekomendasi Customer":
        page_customer_recommendation(df, models)
    elif page == "Admin Input Produk":
        page_admin_input(df, models)
    elif page == "Visualisasi":
        page_visualization(df)
    elif page == "Tentang":
        page_about(models)

if __name__ == "__main__":
    main()