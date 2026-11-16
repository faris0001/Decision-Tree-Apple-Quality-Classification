import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.tree import plot_tree, export_text

# ==========================================
# 1. MEMUAT DATA DAN MODEL
# ==========================================

df = pd.read_csv(
    "dataset/apple_quality_preprocessed.csv"
)

X = df.drop(columns=["Quality"])

model_b = joblib.load(
    "model_decision_tree_B.pkl"
)

fitur = X.columns.tolist()

# ==========================================
# 2. VISUALISASI STRUKTUR DECISION TREE
# ==========================================

plt.figure(figsize=(24, 12))

plot_tree(
    model_b,
    feature_names=fitur,
    class_names=["bad", "good"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title(
    "Visualisasi Decision Tree Model B (Max Depth 4)"
)

plt.tight_layout()

plt.savefig(
    "decision_tree_model_B.png",
    dpi=300
)

plt.close()

# ==========================================
# 3. MENAMPILKAN ATURAN KEPUTUSAN
# ==========================================

print("=" * 60)
print("       ATURAN KEPUTUSAN MODEL B")
print("=" * 60)

aturan = export_text(
    model_b,
    feature_names=fitur
)

print(aturan)

with open(
    "aturan_keputusan.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(aturan)

# ==========================================
# 4. FEATURE IMPORTANCE MODEL B
# ==========================================

importance = pd.DataFrame({
    "Fitur": fitur,
    "Importance": model_b.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- FEATURE IMPORTANCE MODEL B ---")
print(importance.to_string(index=False))

# ==========================================
# 5. VISUALISASI FEATURE IMPORTANCE
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Fitur"
)

plt.title(
    "Feature Importance Decision Tree Model B"
)

plt.xlabel("Tingkat Kepentingan")
plt.ylabel("Fitur")

plt.tight_layout()

plt.savefig(
    "feature_importance_model_B.png",
    dpi=300
)

plt.close()

# ==========================================
# 6. MENYIMPAN HASIL
# ==========================================

importance.to_csv(
    "feature_importance_model_B.csv",
    index=False
)

print("\nFile berhasil dibuat:")
print("1. decision_tree_model_B.png")
print("2. aturan_keputusan.txt")
print("3. feature_importance_model_B.png")
print("4. feature_importance_model_B.csv")

print("\n" + "=" * 60)
print("       INTERPRETASI MODEL SELESAI")
print("=" * 60)