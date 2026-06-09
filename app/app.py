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

# ======================================================
# KONFIGURASI DASAR
# ======================================================
st.set_page_config(
    page_title="Skincare Recommendation System",
    page_icon="🧴",
    layout="wide",
)

st.markdown("""
<style>
/* ====== SIDEBAR WEB STYLE ====== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff1f7 0%, #ffe4ef 45%, #ffffff 100%);
    border-right: 1px solid #f9a8d4;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.sidebar-brand {
    padding: 20px 16px;
    margin-bottom: 18px;
    border-radius: 22px;
    background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);
    color: white;
    box-shadow: 0 12px 30px rgba(190, 24, 93, 0.25);
}

.sidebar-logo {
    font-size: 28px;
    font-weight: 900;
    letter-spacing: -0.6px;
    margin-bottom: 4px;
}

.sidebar-tagline {
    font-size: 13px;
    opacity: 0.92;
    line-height: 1.5;
}

.sidebar-section-title {
    font-size: 12px;
    font-weight: 800;
    color: #9d174d;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 10px 0 8px 4px;
}

.sidebar-footer {
    margin-top: 28px;
    padding: 14px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid #fbcfe8;
    font-size: 12px;
    color: #831843;
    line-height: 1.5;
}

/* Radio menu sidebar */
/* ====== SIDEBAR MENU TANPA BULATAN RADIO ====== */

/* Hilangkan bulatan radio */
section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* Style item menu */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100%;
    padding: 12px 14px !important;
    border-radius: 16px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.55);
    border: 1px solid transparent;
    transition: all 0.25s ease;
    cursor: pointer;
}

/* Hover menu */
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #ffffff;
    border: 1px solid #f9a8d4;
    transform: translateX(4px);
    box-shadow: 0 8px 22px rgba(190, 24, 93, 0.12);
}

/* Text menu */
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-weight: 700;
    color: #831843;
    font-size: 14px;
}

/* Menu aktif */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);
    border: 1px solid #ec4899;
    box-shadow: 0 10px 26px rgba(190, 24, 93, 0.25);
    transform: translateX(4px);
}

/* Text menu aktif */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: white !important;
    font-weight: 800;
}

/* Background utama */
.stApp {
    background: linear-gradient(135deg, #fff7fb 0%, #fdf2f8 45%, #ffffff 100%);
}

/* Container utama */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fce7f3 0%, #fff7fb 100%);
    border-right: 1px solid #f9a8d4;
}

/* Judul */
h1 {
    color: #9d174d;
    font-weight: 800;
}

h2, h3 {
    color: #be185d;
}

/* Metric card */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #fbcfe8;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(190, 24, 93, 0.08);
}

/* Tombol */
.stButton > button {
    background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1.2rem;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(236, 72, 153, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #db2777 0%, #9d174d 100%);
    color: white;
    border: none;
}

/* Input, selectbox, textarea */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {
    border-radius: 14px !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #fbcfe8;
}

/* Alert */
div[data-testid="stAlert"] {
    border-radius: 16px;
}

/* Card custom */
.hero-card {
    padding: 34px;
    border-radius: 26px;
    background: linear-gradient(135deg, #fce7f3 0%, #ffffff 65%);
    border: 1px solid #f9a8d4;
    box-shadow: 0 12px 35px rgba(190, 24, 93, 0.12);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 40px;
    font-weight: 850;
    color: #9d174d;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    color: #831843;
    line-height: 1.6;
}

/* Product card */
.product-card {
    padding: 18px;
    border-radius: 20px;
    background: #ffffff;
    border: 1px solid #fbcfe8;
    box-shadow: 0 8px 26px rgba(190, 24, 93, 0.08);
    margin-bottom: 14px;
}

.product-title {
    color: #9d174d;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 8px;
}

.product-badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background: #fce7f3;
    color: #9d174d;
    font-weight: 700;
    font-size: 12px;
    margin-right: 6px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

PROJECT_TITLE = "Sistem Rekomendasi Produk Skincare Menggunakan Clustering dan Classification"
IDENTITAS_ANGGOTA = [
    "Zakia Ramadhani (24051214081)",
    "Komang Nency Astiti (24051214082)",
]

# Folder dasar. Struktur yang disarankan:
# UAS_DataMining_NamaKelompok/
# ├── app/app.py
# ├── dataset/skincare_clustered.csv
# ├── model/*.pkl
# └── assets/*.png
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent
BASE_DIR = APP_DIR.parent

DATA_CANDIDATES = [
    BASE_DIR / "dataset" / "skincare_clustered.csv",
    BASE_DIR / "skincare_clustered.csv",
    APP_DIR / "dataset" / "skincare_clustered.csv",
    APP_DIR / "skincare_clustered.csv",
]

ASSET_DIR_CANDIDATES = [
    BASE_DIR / "assets",
    APP_DIR / "assets",
]

MODEL_DIR_CANDIDATES = [
    BASE_DIR / "model",
    APP_DIR / "model",
]

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


def is_face_product_series(df):
    """Menentukan produk face skincare secara fleksibel dari kolom is_face_skincare atau product_type."""
    if "is_face_skincare" in df.columns:
        return df["is_face_skincare"].astype(str).str.lower().isin(["true", "1", "yes", "ya"])
    if "product_type" in df.columns:
        return df["product_type"].astype(str).str.lower().isin(FACE_SKINCARE_TYPES)
    return pd.Series([True] * len(df), index=df.index)

# ======================================================
# CSS SEDERHANA UNTUK UI
# ======================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 1rem;
        border: 1px solid #eee;
        background: #fff;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    }
    .product-card {
        padding: 1rem;
        border-radius: 1rem;
        border: 1px solid #ececec;
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 5px rgba(0,0,0,0.04);
    }
    .tag {
        display: inline-block;
        padding: 0.20rem 0.55rem;
        border-radius: 999px;
        background: #f3f3f3;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
        font-size: 0.82rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.75rem;
        background: #fff8e6;
        border: 1px solid #ffe0a3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# FUNGSI UTILITAS
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
    except Exception as exc:
        st.warning(f"File model {filename} ditemukan, tetapi gagal dibaca: {exc}")
        return None


def clean_ingredient_name(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ", "_")
    return text


def parse_ingredients(value):
    """Mengubah kolom ingredients menjadi list meskipun formatnya string list atau koma."""
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
    if price_idr < 150_000:
        return "Low"
    if price_idr <= 500_000:
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

    # Normalisasi kolom penting agar app tidak error.
    if "ingredients_list" not in df.columns:
        if "clean_ingreds" in df.columns:
            df["ingredients_list"] = df["clean_ingreds"].apply(parse_ingredients)
        elif "ingredients_str" in df.columns:
            df["ingredients_list"] = df["ingredients_str"].astype(str).str.replace("_", " ").str.split()
        else:
            df["ingredients_list"] = [[] for _ in range(len(df))]
    else:
        df["ingredients_list"] = df["ingredients_list"].apply(parse_ingredients)

    if "ingredients_str" not in df.columns:
        df["ingredients_str"] = df["ingredients_list"].apply(ingredients_list_to_str)

    if "num_ingredients" not in df.columns:
        df["num_ingredients"] = df["ingredients_list"].apply(len)
    df["num_ingredients"] = pd.to_numeric(df["num_ingredients"], errors="coerce").fillna(0).astype(int)

    if "price_idr" not in df.columns:
        if "price" in df.columns:
            df["price_idr"] = (
                df["price"].astype(str).str.replace("£", "", regex=False).str.strip().astype(float) * 20_000
            )
        else:
            df["price_idr"] = np.nan
    df["price_idr"] = pd.to_numeric(df["price_idr"], errors="coerce")

    if "price_category" not in df.columns:
        df["price_category"] = df["price_idr"].apply(infer_price_category)
    df["price_category"] = df["price_category"].astype(str)

    if "product_type" not in df.columns:
        df["product_type"] = "Unknown"
    df["product_type"] = df["product_type"].astype(str)

    if "product_name" not in df.columns:
        df["product_name"] = "Produk Tanpa Nama"
    df["product_name"] = df["product_name"].astype(str)

    if "cluster_label" not in df.columns:
        if "cluster" in df.columns:
            df["cluster_label"] = df["cluster"].map(DEFAULT_CLUSTER_LABELS)
        else:
            df["cluster_label"] = "Belum diketahui"

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


def show_missing_file_warning(df, models):
    missing = []
    if df.empty:
        missing.append("dataset/skincare_clustered.csv")
    if models["classification_model"] is None:
        missing.append("model/classification_best_model_dataset_terbaru.pkl")
    if models["kmeans_model"] is None:
        missing.append("model/kmeans_model.pkl")
    if models["tfidf_vectorizer"] is None:
        missing.append("model/tfidf_vectorizer.pkl")

    if missing:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.warning(
            "Beberapa file belum ditemukan. Jalankan notebook final terlebih dahulu, lalu pindahkan file hasilnya ke folder dataset/ dan model/."
        )
        st.write("File yang belum ditemukan:")
        for item in missing:
            st.write(f"- `{item}`")
        st.markdown("</div>", unsafe_allow_html=True)


def show_product_card(row):
    ingredients = parse_ingredients(row.get("ingredients_list", []))
    ingredients_preview = ingredients[:8]
    tags = "".join([f"<span class='tag'>{item}</span>" for item in ingredients_preview])
    st.markdown(
        f"""
        <div class='product-card'>
            <h4 style='margin-bottom:0.25rem'>{row.get('product_name', '-')}</h4>
            <p style='margin:0.1rem 0'><b>Jenis Produk:</b> {row.get('product_type', '-')}</p>
            <p style='margin:0.1rem 0'><b>Cluster:</b> {row.get('cluster_label', '-')}</p>
            <p style='margin:0.1rem 0'><b>Kategori Harga:</b> {row.get('price_category', '-')} | <b>Harga:</b> {format_rupiah(row.get('price_idr', np.nan))}</p>
            <p style='margin:0.5rem 0 0.25rem'><b>Ingredients utama:</b></p>
            {tags if tags else '<span class="tag">Tidak tersedia</span>'}
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_new_product_row(product_name, product_type, ingredients_text, price_idr, predicted_label):
    ingredients_list = [item.strip() for item in ingredients_text.split(",") if item.strip()]
    ingredients_str = ingredients_list_to_str(ingredients_list)
    price_category = infer_price_category(price_idr)
    return {
        "product_name": product_name,
        "product_type": product_type,
        "clean_ingreds": str(ingredients_list),
        "ingredients_list": ingredients_list,
        "num_ingredients": len(ingredients_list),
        "ingredients_str": ingredients_str,
        "price_idr": float(price_idr),
        "price_category": price_category,
        "cluster_label": predicted_label,
        "is_face_skincare": str(product_type).lower() in FACE_SKINCARE_TYPES,
    }


def predict_new_product(models, product_type, ingredients_text, price_idr):
    ingredients_list = [item.strip() for item in ingredients_text.split(",") if item.strip()]
    ingredients_str = ingredients_list_to_str(ingredients_list)
    price_category = infer_price_category(price_idr)
    num_ingredients = len(ingredients_list)

    input_df = pd.DataFrame(
        [
            {
                "ingredients_str": ingredients_str,
                "num_ingredients": num_ingredients,
                "price_idr": float(price_idr),
                "price_category": price_category,
                "product_type": product_type,
            }
        ]
    )

    class_prediction = None
    cluster_prediction = None

    if models["classification_model"] is not None:
        try:
            class_prediction = models["classification_model"].predict(input_df)[0]
        except Exception as exc:
            st.warning(f"Model classification gagal melakukan prediksi: {exc}")

    if models["kmeans_model"] is not None and models["tfidf_vectorizer"] is not None:
        try:
            X_new = models["tfidf_vectorizer"].transform([ingredients_str])
            cluster_id = int(models["kmeans_model"].predict(X_new)[0])
            cluster_prediction = models["cluster_labels"].get(cluster_id, f"Cluster {cluster_id}")
        except Exception as exc:
            st.warning(f"Model clustering gagal melakukan prediksi: {exc}")

    final_prediction = class_prediction or cluster_prediction or "Belum dapat diprediksi"
    return final_prediction, class_prediction, cluster_prediction, input_df


def save_dataset_to_csv(df):
    data_path = st.session_state.get("data_path")
    if data_path:
        out_path = Path(data_path)
    else:
        out_path = BASE_DIR / "dataset" / "skincare_clustered.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    st.session_state["data_path"] = str(out_path)
    return out_path


def draw_bar_chart(data, x, y, title):
    if px is not None:
        fig = px.bar(data, x=x, y=y, title=title, text=y)
        fig.update_layout(xaxis_title="", yaxis_title="Jumlah", title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(data.set_index(x)[y])

# ======================================================
# HALAMAN APLIKASI
# ======================================================
def page_home(df, models):
    st.markdown(f"<div class='main-title'>🧴 {PROJECT_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Aplikasi web sederhana untuk merekomendasikan produk berdasarkan tipe kulit, permasalahan kulit, ingredients, jenis produk, dan kategori harga.</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Produk", f"{len(df):,}" if not df.empty else "0")
    col2.metric("Jumlah Jenis Produk", df["product_type"].nunique() if not df.empty else 0)
    col3.metric("Jumlah Cluster", df["cluster_label"].nunique() if not df.empty else 0)
    col4.metric("Model Classification", "Aktif" if models["classification_model"] is not None else "Belum ada")

    st.subheader("Deskripsi Proyek")
    st.write(
        "Proyek ini menerapkan dua metode Data Mining, yaitu **K-Means Clustering** untuk mengelompokkan produk berdasarkan komposisi ingredients dan **Classification** untuk memprediksi kelompok rekomendasi produk baru. Hasil model kemudian diimplementasikan dalam aplikasi web berbasis Streamlit."
    )

    st.subheader("Fitur Utama Aplikasi")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **Customer**
            - Input tipe kulit dan permasalahan kulit.
            - Memilih cakupan rekomendasi: Face atau Face and Body.
            - Memilih kategori harga dan jumlah rekomendasi.
            - Melihat daftar produk yang paling sesuai.

            **Admin**
            - Menambahkan produk skincare/bodycare baru.
            - Sistem memprediksi cluster produk.
            - Produk baru dapat disimpan ke database aktif.
            """
        )
    with col_b:
        st.markdown(
            """
            **Visualisasi**
            - Grafik pendukung dataset.
            - Visualisasi hasil analisis clustering dan classification.

            **Tentang**
            - Penjelasan metode Data Mining.
            - Kumpulan data yang digunakan.
            - Informasi proyek dan alur sistem.
            """
        )

    st.subheader("Identitas Anggota")
    for anggota in IDENTITAS_ANGGOTA:
        st.write(f"- {anggota}")


def page_dataset(df):
    st.title("📊 Gambaran Umum Dataset")
    if df.empty:
        st.error("Dataset belum ditemukan. Letakkan file `skincare_clustered.csv` di folder `dataset/`.")
        return

    st.write(f"Dataset aktif: `{st.session_state.get('data_path')}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", f"{df.shape[0]:,}")
    col2.metric("Jumlah Kolom", f"{df.shape[1]:,}")
    col3.metric("Missing Value", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Preview Dataset")
    st.dataframe(df.head(30), use_container_width=True)

    st.subheader("Statistik Numerik")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    else:
        st.write("Tidak ada kolom numerik yang tersedia.")

    st.subheader("Distribusi Kolom Kategorikal")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Jenis Produk**")
        st.dataframe(df["product_type"].value_counts().reset_index().rename(columns={"index": "product_type", "product_type": "jumlah"}), use_container_width=True)
    with col_b:
        st.write("**Cluster Label**")
        st.dataframe(df["cluster_label"].value_counts().reset_index().rename(columns={"index": "cluster_label", "cluster_label": "jumlah"}), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Dataset Aktif", csv, "skincare_clustered_active.csv", "text/csv")


def page_customer_recommendation(df, models):
    st.title("✨ Prediksi / Rekomendasi Customer")
    if df.empty:
        st.error("Dataset belum ditemukan. Rekomendasi belum dapat dijalankan.")
        return

    mapping = models["skin_concern_mapping"]

    with st.form("form_rekomendasi_customer"):
        col1, col2 = st.columns(2)
        with col1:
            skin_type = st.selectbox("Tipe Kulit", ["Oily", "Dry", "Combination", "Sensitive"])
            concern = st.selectbox("Permasalahan Kulit", ["Acne", "Brightening", "Anti-Aging", "Hydration"])
            product_types = ["Semua"] + sorted(df["product_type"].dropna().astype(str).unique().tolist())
            product_type = st.selectbox("Jenis Produk yang Diinginkan", product_types)
        with col2:
            price_categories = ["Semua"] + sorted(df["price_category"].dropna().astype(str).unique().tolist())
            price_category = st.selectbox("Kategori Harga", price_categories)
            product_scope = st.selectbox(
                "Cakupan Rekomendasi",
                ["Face (Skincare)", "Face and Body (Skincare + Bodycare)"],
            )
            top_n = st.slider("Jumlah Rekomendasi", min_value=3, max_value=20, value=10)

        sort_by = st.selectbox(
            "Urutkan Berdasarkan",
            ["Harga termurah", "Harga termahal", "Jumlah ingredients terbanyak", "Acak"],
        )
        submitted = st.form_submit_button("Proses Rekomendasi")

    if submitted:
        target_label = mapping.get((skin_type, concern), "Hydrating & Anti-Aging")
        result = df[df["cluster_label"] == target_label].copy()

        if product_type != "Semua":
            result = result[result["product_type"].astype(str) == product_type]
        if price_category != "Semua":
            result = result[result["price_category"].astype(str) == price_category]
        if product_scope == "Face (Skincare)":
            result = result[is_face_product_series(result)]

        if sort_by == "Harga termurah":
            result = result.sort_values("price_idr", ascending=True)
        elif sort_by == "Harga termahal":
            result = result.sort_values("price_idr", ascending=False)
        elif sort_by == "Jumlah ingredients terbanyak":
            result = result.sort_values("num_ingredients", ascending=False)
        else:
            result = result.sample(frac=1, random_state=42)

        result = result.head(top_n)

        st.success(f"Rekomendasi cluster untuk kulit **{skin_type}** dengan concern **{concern}** adalah **{target_label}**. Cakupan rekomendasi: **{product_scope}**.")
        st.write(f"Jumlah produk yang cocok setelah filter: **{len(result)} produk**")

        if result.empty:
            st.warning("Tidak ada produk yang cocok dengan filter tersebut. Coba pilih jenis produk atau harga 'Semua'.")
            return

        for _, row in result.iterrows():
            show_product_card(row)

        st.download_button(
            "Download Hasil Rekomendasi",
            result.to_csv(index=False).encode("utf-8"),
            "hasil_rekomendasi_skincare.csv",
            "text/csv",
        )


def page_admin_input(df, models):
    st.title("🛠️ Admin: Input Produk Baru")
    st.write(
        "Halaman ini digunakan untuk menambahkan produk baru. Sistem akan memprediksi kelompok produk menggunakan model classification. Jika model classification belum tersedia, sistem mencoba menggunakan model clustering."
    )

    existing_types = sorted(df["product_type"].dropna().astype(str).unique().tolist()) if not df.empty and "product_type" in df.columns else []
    default_types = sorted(set(existing_types + ["serum", "cleanser", "moisturiser", "toner", "mask", "eye cream", "essence", "body wash", "sunscreen"]))

    with st.form("form_input_produk"):
        product_name = st.text_input("Nama Produk", placeholder="Contoh: Glow Brightening Serum")
        product_type = st.selectbox("Jenis Produk", default_types if default_types else ["serum", "cleanser", "moisturiser"])
        ingredients_text = st.text_area(
            "Ingredients",
            placeholder="Tulis ingredients dipisahkan koma. Contoh: niacinamide, glycerin, sodium hyaluronate, panthenol",
            height=120,
        )
        price_idr = st.number_input("Harga Produk (IDR)", min_value=0, value=150000, step=5000)
        submitted = st.form_submit_button("Prediksi Cluster Produk")

    if submitted:
        if not product_name.strip() or not ingredients_text.strip():
            st.error("Nama produk dan ingredients wajib diisi.")
            return

        final_prediction, class_prediction, cluster_prediction, input_df = predict_new_product(
            models=models,
            product_type=product_type,
            ingredients_text=ingredients_text,
            price_idr=price_idr,
        )

        st.subheader("Hasil Prediksi")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediksi Akhir", final_prediction)
        col2.metric("Classification", class_prediction if class_prediction is not None else "Tidak tersedia")
        col3.metric("Clustering", cluster_prediction if cluster_prediction is not None else "Tidak tersedia")

        st.write("**Data input yang masuk ke model classification:**")
        st.dataframe(input_df, use_container_width=True)

        new_row = make_new_product_row(product_name, product_type, ingredients_text, price_idr, final_prediction)
        st.session_state["new_product_row"] = new_row

    if "new_product_row" in st.session_state:
        st.divider()
        st.subheader("Simpan Produk Baru")
        st.write("Produk baru dapat disimpan ke dataset aktif agar muncul pada rekomendasi berikutnya.")
        if st.button("Simpan ke Dataset Aktif"):
            new_row = st.session_state["new_product_row"]
            new_df = pd.DataFrame([new_row])
            combined = pd.concat([st.session_state["data"], new_df], ignore_index=True)
            st.session_state["data"] = combined
            out_path = save_dataset_to_csv(combined)
            st.success(f"Produk berhasil disimpan ke `{out_path}`.")


def page_visualization(df):
    st.title("📈 Visualisasi")
    st.write(
        "Halaman ini berisi **grafik pendukung** dan **visualisasi hasil analisis** untuk membantu membaca pola dataset, distribusi harga, jenis produk, serta hasil clustering."
    )
    st.markdown(
        """
        **Isi halaman Visualisasi:**
        - Grafik pendukung dataset produk skincare dan bodycare.
        - Visualisasi hasil analisis clustering, seperti distribusi cluster dan rata-rata harga per cluster.
        - Visualisasi tambahan dari notebook apabila file gambar PNG sudah dipindahkan ke folder `assets/`.
        """
    )
    if df.empty:
        st.error("Dataset belum ditemukan. Visualisasi belum dapat ditampilkan.")
        return

    tab1, tab2 = st.tabs(["Grafik Pendukung", "Visualisasi Hasil Analisis"])

    with tab1:
        st.subheader("Distribusi Kategori Harga")
        price_dist = df["price_category"].value_counts().reset_index()
        price_dist.columns = ["price_category", "jumlah"]
        draw_bar_chart(price_dist, "price_category", "jumlah", "Distribusi Kategori Harga")

        st.subheader("Distribusi Jenis Produk")
        product_dist = df["product_type"].value_counts().head(20).reset_index()
        product_dist.columns = ["product_type", "jumlah"]
        draw_bar_chart(product_dist, "product_type", "jumlah", "Distribusi Jenis Produk")

        st.subheader("Distribusi Cluster")
        cluster_dist = df["cluster_label"].value_counts().reset_index()
        cluster_dist.columns = ["cluster_label", "jumlah"]
        draw_bar_chart(cluster_dist, "cluster_label", "jumlah", "Distribusi Cluster Label")

        if "price_idr" in df.columns:
            st.subheader("Rata-rata Harga per Cluster")
            avg_price = df.groupby("cluster_label", as_index=False)["price_idr"].mean()
            avg_price["price_idr"] = avg_price["price_idr"].round(0)
            draw_bar_chart(avg_price, "cluster_label", "price_idr", "Rata-rata Harga per Cluster")

    with tab2:
        st.write("Bagian ini akan menampilkan gambar hasil visualisasi yang sudah disimpan dari notebook, jika file PNG tersedia di folder `assets/`.")
        asset_files = [
            "price_category_distribution.png",
            "product_type_distribution.png",
            "elbow_silhouette.png",
            "cluster_pca.png",
            "cluster_actives.png",
            "cluster_product_type.png",
        ]
        found_any = False
        for filename in asset_files:
            asset_path = find_asset_path(filename)
            if asset_path is not None:
                found_any = True
                st.image(str(asset_path), caption=filename, use_container_width=True)
        if not found_any:
            st.info("Belum ada file gambar dari notebook. Jalankan notebook sampai bagian visualisasi, lalu pindahkan file PNG ke folder `assets/`.")


def page_about(models):
    st.title("ℹ️ Tentang Proyek")
    st.write(
        "Halaman ini menjelaskan **metode yang digunakan**, **kumpulan data**, dan **informasi proyek** agar aplikasi mudah dipahami saat presentasi UAS."
    )

    st.subheader("Informasi Proyek")
    st.write(
        "Proyek ini membangun sistem rekomendasi produk skincare dan bodycare berbasis Data Mining. Aplikasi memiliki dua pengguna utama, yaitu customer untuk mendapatkan rekomendasi produk dan admin untuk menambahkan produk baru ke database."
    )
    st.write(
        "Framework kerja yang digunakan mengikuti alur **CRISP-DM**, mulai dari pemahaman masalah, pemahaman data, preprocessing, pemodelan, evaluasi, sampai implementasi model dalam aplikasi web."
    )

    st.subheader("Kumpulan Data")
    st.write(
        "Dataset utama adalah `skincare_clustered.csv`, yaitu dataset hasil preprocessing dan clustering dari notebook. Dataset berisi informasi produk seperti nama produk, jenis produk, daftar ingredients, jumlah ingredients, harga, kategori harga, dan label cluster."
    )
    st.write(
        "Produk baru yang dimasukkan admin akan disimpan kembali ke dataset aktif sehingga dapat digunakan sebagai database rekomendasi pada proses berikutnya."
    )

    st.subheader("Penjelasan Metode")
    st.write(
        "**Clustering** dilakukan menggunakan algoritma K-Means dengan fitur TF-IDF dari ingredients. Tujuannya adalah mengelompokkan produk berdasarkan kemiripan komposisi bahan. Label cluster yang digunakan adalah Exfoliating & Brightening, Hydrating & Anti-Aging, serta Acne & Purifying."
    )
    st.write(
        "**Classification** dilakukan menggunakan pipeline preprocessing dan model terbaik dari perbandingan Decision Tree dan Random Forest. Fitur yang digunakan adalah `ingredients_str`, `num_ingredients`, `price_idr`, `price_category`, dan `product_type`, sedangkan targetnya adalah `cluster_label`."
    )

    st.subheader("Alur Sistem")
    st.write(
        "1. Dataset skincare dibersihkan dan harga dikonversi ke Rupiah.\n"
        "2. Produk dikelompokkan menggunakan K-Means berdasarkan ingredients.\n"
        "3. Label cluster digunakan sebagai target model classification.\n"
        "4. Customer memasukkan tipe kulit dan concern. Sistem menentukan cluster rekomendasi.\n"
        "5. Admin dapat memasukkan produk baru. Sistem memprediksi cluster produk tersebut."
    )

    st.subheader("Status File Model")
    status = pd.DataFrame(
        [
            {"File": "classification_best_model_dataset_terbaru.pkl", "Status": "Ada" if models["classification_model"] is not None else "Belum ada"},
            {"File": "kmeans_model.pkl", "Status": "Ada" if models["kmeans_model"] is not None else "Belum ada"},
            {"File": "tfidf_vectorizer.pkl", "Status": "Ada" if models["tfidf_vectorizer"] is not None else "Belum ada"},
            {"File": "cluster_labels.pkl", "Status": "Ada" if models["cluster_labels"] is not None else "Default"},
            {"File": "skin_concern_mapping.pkl", "Status": "Ada" if models["skin_concern_mapping"] is not None else "Default"},
        ]
    )
    st.dataframe(status, use_container_width=True)

    st.subheader("Catatan Penggunaan")
    st.code(
        """# Jalankan dari folder utama proyek:
streamlit run app/app.py

# File yang perlu ada:
dataset/skincare_clustered.csv
model/classification_best_model_dataset_terbaru.pkl
model/kmeans_model.pkl
model/tfidf_vectorizer.pkl
model/cluster_labels.pkl
model/skin_concern_mapping.pkl
assets/*.png  # opsional untuk visualisasi dari notebook
""",
        language="bash",
    )


def main():
    init_session_data()
    df = st.session_state["data"]
    models = load_models_cached()

    st.sidebar.title("🧴 Menu Aplikasi")
    page = st.sidebar.radio(
        "Pilih Halaman",
        [
            "Beranda",
            "Gambaran Umum Dataset",
            "Prediksi / Rekomendasi Customer",
            "Admin Input Produk",
            "Visualisasi",
            "Tentang",
        ],
    )

    st.sidebar.divider()

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
