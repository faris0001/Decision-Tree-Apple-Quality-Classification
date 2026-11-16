# ==========================================
# TP-05 PEMBELAJARAN MESIN
# VALIDASI DAN PEMBERSIHAN DATASET
# KLASIFIKASI KUALITAS BUAH APEL
# ==========================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Membaca Dataset
df = pd.read_csv("dataset/apple_quality.csv")

print("===================================")
print("   VALIDASI DATASET APPLE QUALITY")
print("===================================")

# 2. Menampilkan Data Awal
print("\n--- 5 Data Pertama ---")
print(df.head())

# 3. Informasi Dataset Awal
print("\n--- Informasi Dataset Awal ---")
print(df.info())

# 4. Membersihkan Kolom Acidity
df['Acidity'] = pd.to_numeric(
    df['Acidity'],
    errors='coerce'
)

# 5. Menghapus Baris yang Tidak Valid
df = df.dropna()

# 6. Menghapus Kolom ID
df = df.drop(columns=['A_id'])

# 7. Memastikan Tipe Data
print("\n--- Informasi Dataset Setelah Dibersihkan ---")
print(df.info())

# 8. Memeriksa Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 9. Ukuran Dataset
print("\n--- Ukuran Dataset Bersih ---")
print("Jumlah Baris :", df.shape[0])
print("Jumlah Kolom :", df.shape[1])

# 10. Distribusi Kualitas Apel
print("\n--- Distribusi Kualitas Apel ---")
print(df['Quality'].value_counts())

# 11. Menampilkan Statistik Deskriptif
print("\n--- Statistik Deskriptif ---")
print(df.describe())

# 12. Visualisasi Distribusi Kualitas
plt.figure(figsize=(8, 5))

sns.countplot(
    x='Quality',
    data=df
)

plt.title('Distribusi Kualitas Buah Apel')
plt.xlabel('Kualitas Apel')
plt.ylabel('Jumlah Data')

plt.tight_layout()

plt.savefig("distribusi_kualitas_apel.png")

plt.show()

# 13. Menyimpan Dataset Bersih
df.to_csv(
    "dataset/apple_quality_clean.csv",
    index=False
)

print("\nDataset berhasil dibersihkan!")
print("Dataset bersih disimpan!")

print("\nVALIDASI SELESAI!")