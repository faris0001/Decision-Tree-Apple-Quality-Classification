import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. MEMUAT DATASET
# ==========================================

print("=" * 55)
print("       EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 55)

df = pd.read_csv("dataset/apple_quality_clean.csv")

print("\n--- Informasi Dataset ---")
print("Jumlah Baris:", df.shape[0])
print("Jumlah Kolom:", df.shape[1])

print("\n--- Lima Data Pertama ---")
print(df.head())

print("\n--- Informasi Tipe Data ---")
print(df.info())

# ==========================================
# 2. PEMERIKSAAN MISSING VALUES
# ==========================================

print("\n--- Pemeriksaan Missing Values ---")

missing = df.isnull().sum()
print(missing)

print("\nTotal Missing Values:", missing.sum())

# ==========================================
# 3. PENANGANAN MISSING VALUES
# ==========================================

print("\n--- Penanganan Missing Values ---")

for kolom in df.columns:

    if df[kolom].isnull().sum() > 0:

        if df[kolom].dtype == "object":
            df[kolom] = df[kolom].fillna(
                df[kolom].mode()[0]
            )

        else:
            df[kolom] = df[kolom].fillna(
                df[kolom].median()
            )

print("Missing values setelah penanganan:")
print(df.isnull().sum().sum())

# ==========================================
# 4. ANALISIS DISTRIBUSI TARGET
# ==========================================

print("\n--- Distribusi Kualitas Apel ---")
print(df["Quality"].value_counts())

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Quality"
)

plt.title("Distribusi Kualitas Buah Apel")
plt.xlabel("Kualitas Apel")
plt.ylabel("Jumlah Data")

plt.tight_layout()
plt.savefig("eda_distribusi_target.png", dpi=300)
plt.close()

# ==========================================
# 5. ANALISIS HUBUNGAN FITUR DAN TARGET
# ==========================================

fitur = [
    "Size",
    "Weight",
    "Sweetness",
    "Crunchiness",
    "Juiciness",
    "Ripeness",
    "Acidity"
]

fig, axes = plt.subplots(
    4, 2,
    figsize=(14, 18)
)

axes = axes.flatten()

for i, kolom in enumerate(fitur):

    sns.boxplot(
        data=df,
        x="Quality",
        y=kolom,
        ax=axes[i]
    )

    axes[i].set_title(
        f"Hubungan {kolom} dengan Quality"
    )

axes[-1].axis("off")

plt.tight_layout()
plt.savefig("eda_hubungan_fitur_target.png", dpi=300)
plt.close()

# ==========================================
# 6. ANALISIS KORELASI
# ==========================================

df_korelasi = df.copy()

df_korelasi["Quality"] = df_korelasi[
    "Quality"
].map({
    "bad": 0,
    "good": 1
})

plt.figure(figsize=(11, 8))

sns.heatmap(
    df_korelasi.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Heatmap Korelasi Fitur dan Kualitas Apel")

plt.tight_layout()
plt.savefig("eda_heatmap_korelasi.png", dpi=300)
plt.close()

# ==========================================
# 7. PREPROCESSING FITUR KATEGORIKAL
# ==========================================

print("\n--- Preprocessing Data Kategorikal ---")

X = df.drop(columns=["Quality"])
y = df["Quality"].copy()

# Konversi fitur kategorikal jika ditemukan
X = pd.get_dummies(X, drop_first=True)

# Encoding target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

df_preprocessed = X.copy()
df_preprocessed["Quality"] = y_encoded

df_preprocessed.to_csv(
    "dataset/apple_quality_preprocessed.csv",
    index=False
)

print("Encoding target:")
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

print("\nFitur setelah preprocessing:")
print(X.columns.tolist())

print("\nJumlah fitur:", X.shape[1])

# ==========================================
# 8. MENYIMPAN HASIL
# ==========================================

print("\nDataset hasil preprocessing berhasil disimpan.")

print("\n" + "=" * 55)
print("             EDA SELESAI")
print("=" * 55)

print("\nFile visualisasi yang dihasilkan:")
print("1. eda_distribusi_target.png")
print("2. eda_hubungan_fitur_target.png")
print("3. eda_heatmap_korelasi.png")
print("4. dataset/apple_quality_preprocessed.csv")