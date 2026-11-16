import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

print("=" * 50)
print("   PENGUJIAN DECISION TREE PADA DATA UJI")
print("=" * 50)

# Membaca dataset
df = pd.read_csv("dataset/apple_quality_clean.csv")

X = df.drop("Quality", axis=1)
y = df["Quality"]

# Membagi dataset dengan pengaturan yang sama
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Memuat model
model = joblib.load("model_decision_tree.pkl")

# Mengambil 10 sampel khusus dari data uji
sampel = X_test.sample(10, random_state=42)
y_aktual = y_test.loc[sampel.index]

# Melakukan prediksi
y_prediksi = model.predict(sampel)

# Membandingkan hasil
hasil = pd.DataFrame({
    "Kualitas Aktual": y_aktual.values,
    "Hasil Prediksi": y_prediksi
})

hasil["Status"] = [
    "Benar" if aktual == prediksi else "Salah"
    for aktual, prediksi in zip(y_aktual.values, y_prediksi)
]

print("\n--- HASIL PENGUJIAN ---")
print(hasil.to_string(index=False))

benar = (hasil["Status"] == "Benar").sum()
total = len(hasil)

print("\n--- RINGKASAN ---")
print("Total sampel    :", total)
print("Prediksi benar  :", benar)
print("Prediksi salah  :", total - benar)
print(f"Persentase benar: {benar / total * 100:.2f}%")

print("\nPENGUJIAN SELESAI!")