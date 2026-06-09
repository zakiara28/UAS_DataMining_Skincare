# UAS Data Mining - Web Skincare Recommendation System

Template aplikasi web Streamlit untuk proyek UAS Data Mining dengan kombinasi metode **Clustering** dan **Classification**.

## Fitur Web

1. **Beranda**
   - Judul proyek
   - Deskripsi singkat proyek
   - Identitas anggota
   - Ringkasan fitur customer, admin, visualisasi, dan tentang

2. **Gambaran Umum Dataset**
   - Informasi dataset
   - Jumlah data dan jumlah kolom
   - Statistik sederhana
   - Preview dataset aktif

3. **Prediksi / Rekomendasi Customer**
   - Customer mengisi tipe kulit dan permasalahan kulit
   - Customer memilih cakupan rekomendasi: Face atau Face and Body
   - Customer memilih kategori harga dan jumlah rekomendasi
   - Sistem menampilkan produk yang sesuai berdasarkan cluster rekomendasi

4. **Admin Input Produk**
   - Admin menambahkan produk skincare/bodycare baru
   - Sistem memprediksi cluster produk baru menggunakan model classification atau clustering
   - Produk baru dapat disimpan ke database aktif agar ikut digunakan pada rekomendasi berikutnya

5. **Visualisasi**
   - Grafik pendukung dataset
   - Visualisasi hasil analisis clustering dan harga
   - Menampilkan visualisasi dari notebook jika file PNG tersedia di folder `assets/`

6. **Tentang**
   - Penjelasan metode
   - Kumpulan data
   - Informasi proyek dan alur sistem

## Struktur Folder

```text
UAS_DataMining_Web_Skincare/
├── app/
│   └── app.py
├── dataset/
│   └── skincare_clustered.csv
├── model/
│   ├── classification_best_model_dataset_terbaru.pkl
│   ├── kmeans_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── cluster_labels.pkl
│   └── skin_concern_mapping.pkl
├── assets/
│   └── file_visualisasi_dari_notebook.png
├── requirements.txt
└── README.md
```

## Cara Menjalankan

1. Jalankan notebook final sampai menghasilkan file dataset dan model.
2. Pindahkan file `skincare_clustered.csv` ke folder `dataset/`.
3. Pindahkan file `.pkl` ke folder `model/`.
4. Pindahkan file grafik `.png` dari notebook ke folder `assets/` jika ada.
5. Jalankan perintah berikut dari folder utama:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Catatan

Ubah bagian `IDENTITAS_ANGGOTA` di `app/app.py` sesuai nama dan NIM kelompok.
# UAS_DataMining_Skincare
