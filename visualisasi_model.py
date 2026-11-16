import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree

print("=" * 45)
print("     VISUALISASI MODEL DECISION TREE")
print("=" * 45)

# Membaca dataset
df = pd.read_csv("dataset/apple_quality_clean.csv")

X = df.drop("Quality", axis=1)
y = df["Quality"]

# Pembagian dataset sama seperti pelatihan
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Memuat model
model = joblib.load("model_decision_tree.pkl")

# Prediksi
y_pred = model.predict(X_test)

# 1. Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["bad", "good"])

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Bad", "Good"],
    yticklabels=["Bad", "Good"]
)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("\nConfusion Matrix berhasil disimpan!")

# 2. Visualisasi Struktur Decision Tree
plt.figure(figsize=(24, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["bad", "good"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Struktur Pohon Keputusan - Decision Tree")
plt.savefig("struktur_decision_tree.png", dpi=300)
plt.close()

print("Struktur Decision Tree berhasil disimpan!")

print("\nVISUALISASI SELESAI!")