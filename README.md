# Klasifikasi Kualitas Buah Apel Menggunakan Algoritma Decision Tree

## Deskripsi Proyek

Proyek ini merupakan implementasi algoritma Machine Learning Decision Tree untuk mengklasifikasikan kualitas buah apel berdasarkan karakteristik fisiknya.

Model dikembangkan menggunakan dataset Apple Quality dari Kaggle. Sistem memanfaatkan tujuh fitur untuk memprediksi apakah apel termasuk dalam kategori kualitas baik (GOOD) atau buruk (BAD).

Proyek ini dibuat untuk memenuhi Tugas Akhir 05 (TA-05) mata kuliah Permodelan dan Simulasi.

## Identitas

- Nama: Muhammad Faris
- NIM: 301240055
- Program Studi: Teknik Informatika
- Fakultas: Teknologi Informasi
- Universitas: Universitas Bale Bandung

## Dataset

Dataset yang digunakan adalah Apple Quality Dataset.

Sumber dataset:
https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality

Dataset memiliki 4.000 data apel yang digunakan setelah proses pembersihan.

### Fitur yang Digunakan

| No. | Fitur | Keterangan |
|---|---|---|
| 1 | Size | Ukuran apel |
| 2 | Weight | Berat apel |
| 3 | Sweetness | Tingkat kemanisan |
| 4 | Crunchiness | Tingkat kerenyahan |
| 5 | Juiciness | Tingkat kandungan sari buah |
| 6 | Ripeness | Tingkat kematangan |
| 7 | Acidity | Tingkat keasaman |

Target klasifikasi:
- GOOD: Kualitas apel baik.
- BAD: Kualitas apel buruk.

## Metode

Algoritma yang digunakan adalah Decision Tree Classifier dengan kriteria pemisahan Entropy.

Dalam pengujian dilakukan perbandingan dua model:

- Model A: Decision Tree tanpa pembatasan kedalaman (`max_depth=None`).
- Model B: Decision Tree dengan pembatasan kedalaman (`max_depth=4`).

### Hasil Evaluasi Model

| Parameter | Model A | Model B |
|---|---:|---:|
| Akurasi Training | 100% | 77,91% |
| Akurasi Testing | 83,25% | 76,25% |

Model A memperoleh akurasi pengujian lebih tinggi pada pembagian data yang digunakan. Namun, terdapat selisih performa training dan testing yang menunjukkan indikasi overfitting.

Model B memiliki selisih akurasi training dan testing lebih kecil, tetapi akurasi pengujiannya juga lebih rendah.

Model A dipilih sebagai model operasional berdasarkan hasil akurasi testing pada eksperimen ini.

## Teknologi yang Digunakan

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Flask
- HTML dan CSS

## Struktur Folder

```text
TP05_Decision_Tree/
│
├── dataset/
│   ├── apple_quality.csv
│   ├── apple_quality_clean.csv
│   └── apple_quality_preprocessed.csv
│
├── templates/
│   └── index.html
│
├── app.py
├── eda_preprocessing.py
├── analisis_fitur.py
├── latih_model.py
├── uji_model.py
├── perbandingan_model.py
├── interpretasi_model.py
├── simpan_model.py
├── validasi_dataset.py
├── visualisasi_boxplot.py
│
├── model_decision_tree_A.pkl
├── model_decision_tree_B.pkl
├── model_terbaik.joblib
│
├── confusion_matrix.png
├── perbandingan_akurasi.png
├── perbandingan_confusion_matrix.png
├── decision_tree_model_B.png
├── feature_importance_model_B.png
├── boxplot_hubungan_fitur_quality.png
│
├── requirements.txt
└── README.md
```

## Instalasi dan Menjalankan Proyek

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/TP05_Decision_Tree.git
```

Ganti `USERNAME` dengan username GitHub milik Anda.

### 2. Masuk ke Folder Proyek

```bash
cd TP05_Decision_Tree
```

### 3. Buat Virtual Environment

```bash
python -m venv venv
```

Aktifkan virtual environment:

Windows:

```bash
venv\Scripts\activate
```

### 4. Instalasi Library

```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi Flask

```bash
python app.py
```

Buka browser dan akses:

```text
http://127.0.0.1:5000
```

## Cara Menggunakan Aplikasi

1. Jalankan aplikasi Flask.
2. Buka halaman aplikasi melalui browser.
3. Masukkan tujuh karakteristik apel pada form yang tersedia.
4. Klik tombol prediksi.
5. Sistem akan memproses data menggunakan model Decision Tree.
6. Hasil prediksi akan menampilkan kategori GOOD atau BAD.

## Hasil Proyek

Aplikasi ini diharapkan dapat membantu pengguna melakukan klasifikasi kualitas apel berdasarkan karakteristik fisiknya menggunakan algoritma Decision Tree.

Selain implementasi aplikasi, proyek ini juga mencakup:
- Exploratory Data Analysis (EDA).
- Preprocessing dataset.
- Perbandingan dua model Decision Tree.
- Evaluasi classification report dan confusion matrix.
- Visualisasi struktur pohon keputusan.
- Analisis feature importance.
- Implementasi aplikasi web menggunakan Flask.

## Pengembangan

Proyek ini dapat dikembangkan lebih lanjut dengan melakukan optimasi hyperparameter, cross-validation, serta pengujian menggunakan dataset tambahan untuk meningkatkan kemampuan generalisasi model.

## Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan penyelesaian tugas akademik."# Decision-Tree-Apple-Quality-Classification" 
