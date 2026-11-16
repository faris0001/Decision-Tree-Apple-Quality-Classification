import pandas as pd
import matplotlib.pyplot as plt
import joblib

print("=" * 45)
print("       ANALISIS FEATURE IMPORTANCE")
print("=" * 45)

# Membaca dataset
df = pd.read_csv("dataset/apple_quality_clean.csv")

# Memisahkan fitur
X = df.drop("Quality", axis=1)

# Memuat model
model = joblib.load("model_decision_tree.pkl")

# Mengambil nilai feature importance
importance = model.feature_importances_

# Membuat tabel
hasil = pd.DataFrame({
    "Fitur": X.columns,
    "Importance": importance
})

hasil = hasil.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- Peringkat Pengaruh Fitur ---")
print(hasil.to_string(index=False))

# Membuat visualisasi
plt.figure(figsize=(9, 6))

plt.barh(
    hasil["Fitur"],
    hasil["Importance"]
)

plt.xlabel("Nilai Importance")
plt.ylabel("Fitur")
plt.title("Feature Importance - Decision Tree")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("\nVisualisasi berhasil disimpan!")
print("Nama file: feature_importance.png")

print("\nANALISIS SELESAI!")