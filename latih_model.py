import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 45)
print("     PELATIHAN MODEL DECISION TREE")
print("=" * 45)

# 1. Membaca dataset bersih
df = pd.read_csv("dataset/apple_quality_clean.csv")

# 2. Memisahkan fitur dan target
X = df.drop("Quality", axis=1)
y = df["Quality"]

print("\nJumlah data:", len(df))
print("Jumlah fitur:", X.shape[1])
print("Target klasifikasi:", y.unique())

# 3. Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Pembagian Dataset ---")
print("Data Latih:", len(X_train))
print("Data Uji  :", len(X_test))

# 4. Membuat model Decision Tree
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

# 5. Melatih model
model.fit(X_train, y_train)

print("\nModel Decision Tree berhasil dilatih!")

# 6. Melakukan prediksi
y_pred = model.predict(X_test)

# 7. Evaluasi model
akurasi = accuracy_score(y_test, y_pred)

print("\n--- HASIL EVALUASI MODEL ---")
print(f"Akurasi Model: {akurasi * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# 8. Menyimpan model
joblib.dump(model, "model_decision_tree.pkl")

print("\nModel berhasil disimpan!")
print("Nama file: model_decision_tree.pkl")

print("\nPELATIHAN MODEL SELESAI!")