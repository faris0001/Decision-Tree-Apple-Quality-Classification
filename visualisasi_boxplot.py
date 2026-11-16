import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
df = pd.read_csv("dataset/apple_quality_clean.csv")

# Daftar fitur
fitur = [
    "Size",
    "Weight",
    "Sweetness",
    "Crunchiness",
    "Juiciness",
    "Ripeness",
    "Acidity"
]

# Membuat visualisasi
fig, axes = plt.subplots(4, 2, figsize=(14, 18))

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

    axes[i].set_xlabel("Kualitas Apel")
    axes[i].set_ylabel(kolom)

# Menghapus subplot kosong
fig.delaxes(axes[7])

plt.suptitle(
    "Boxplot Hubungan Karakteristik Apel dengan Quality",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "boxplot_hubungan_fitur_quality.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Boxplot berhasil disimpan!")