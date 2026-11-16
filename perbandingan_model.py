import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. MEMUAT DATASET
# ==========================================

print("=" * 60)
print("       PERBANDINGAN MODEL DECISION TREE")
print("=" * 60)

df = pd.read_csv(
    "dataset/apple_quality_preprocessed.csv"
)

X = df.drop(columns=["Quality"])
y = df["Quality"]

# ==========================================
# 2. MEMBAGI DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Pembagian Dataset ---")
print("Data Training :", len(X_train))
print("Data Testing  :", len(X_test))

# ==========================================
# 3. MEMBANGUN MODEL A
# ==========================================

print("\n--- MODEL A: DECISION TREE TANPA BATAS ---")

model_a = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=None,
    random_state=42
)

model_a.fit(X_train, y_train)

pred_train_a = model_a.predict(X_train)
pred_test_a = model_a.predict(X_test)

akurasi_train_a = accuracy_score(
    y_train, pred_train_a
)

akurasi_test_a = accuracy_score(
    y_test, pred_test_a
)

print("Akurasi Training:", round(akurasi_train_a, 4))
print("Akurasi Testing :", round(akurasi_test_a, 4))

print("\nClassification Report Model A:")
print(
    classification_report(
        y_test,
        pred_test_a,
        target_names=["bad", "good"]
    )
)

print("Confusion Matrix Model A:")
cm_a = confusion_matrix(y_test, pred_test_a)
print(cm_a)

# ==========================================
# 4. MEMBANGUN MODEL B
# ==========================================

print("\n--- MODEL B: DECISION TREE PRUNING ---")

model_b = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

model_b.fit(X_train, y_train)

pred_train_b = model_b.predict(X_train)
pred_test_b = model_b.predict(X_test)

akurasi_train_b = accuracy_score(
    y_train, pred_train_b
)

akurasi_test_b = accuracy_score(
    y_test, pred_test_b
)

print("Akurasi Training:", round(akurasi_train_b, 4))
print("Akurasi Testing :", round(akurasi_test_b, 4))

print("\nClassification Report Model B:")
print(
    classification_report(
        y_test,
        pred_test_b,
        target_names=["bad", "good"]
    )
)

print("Confusion Matrix Model B:")
cm_b = confusion_matrix(y_test, pred_test_b)
print(cm_b)

# ==========================================
# 5. PERBANDINGAN KEDUA MODEL
# ==========================================

hasil = pd.DataFrame({
    "Model": [
        "Model A (Tanpa Batas)",
        "Model B (Max Depth 4)"
    ],
    "Akurasi Training": [
        akurasi_train_a,
        akurasi_train_b
    ],
    "Akurasi Testing": [
        akurasi_test_a,
        akurasi_test_b
    ]
})

print("\n--- TABEL PERBANDINGAN MODEL ---")
print(hasil.to_string(index=False))

hasil.to_csv(
    "hasil_perbandingan_model.csv",
    index=False
)

# ==========================================
# 6. VISUALISASI PERBANDINGAN
# ==========================================

hasil_plot = hasil.melt(
    id_vars="Model",
    var_name="Jenis Akurasi",
    value_name="Akurasi"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=hasil_plot,
    x="Model",
    y="Akurasi",
    hue="Jenis Akurasi"
)

plt.title("Perbandingan Akurasi Model Decision Tree")
plt.ylim(0, 1)
plt.ylabel("Akurasi")
plt.xlabel("Model")
plt.tight_layout()

plt.savefig(
    "perbandingan_akurasi.png",
    dpi=300
)

plt.close()

# ==========================================
# 7. VISUALISASI CONFUSION MATRIX
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    cm_a,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["bad", "good"],
    yticklabels=["bad", "good"],
    ax=axes[0]
)

axes[0].set_title("Confusion Matrix Model A")
axes[0].set_xlabel("Prediksi")
axes[0].set_ylabel("Aktual")

sns.heatmap(
    cm_b,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["bad", "good"],
    yticklabels=["bad", "good"],
    ax=axes[1]
)

axes[1].set_title("Confusion Matrix Model B")
axes[1].set_xlabel("Prediksi")
axes[1].set_ylabel("Aktual")

plt.tight_layout()

plt.savefig(
    "perbandingan_confusion_matrix.png",
    dpi=300
)

plt.close()

# ==========================================
# 8. MENYIMPAN MODEL
# ==========================================

joblib.dump(
    model_a,
    "model_decision_tree_A.pkl"
)

joblib.dump(
    model_b,
    "model_decision_tree_B.pkl"
)

print("\nKedua model berhasil disimpan.")

print("\nFile yang dihasilkan:")
print("1. hasil_perbandingan_model.csv")
print("2. perbandingan_akurasi.png")
print("3. perbandingan_confusion_matrix.png")
print("4. model_decision_tree_A.pkl")
print("5. model_decision_tree_B.pkl")

print("\n" + "=" * 60)
print("       PERBANDINGAN MODEL SELESAI")
print("=" * 60)