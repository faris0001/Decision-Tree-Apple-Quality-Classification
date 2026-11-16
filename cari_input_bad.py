import pandas as pd

df = pd.read_csv("dataset/apple_quality_clean.csv")

# Ambil satu contoh data apel dengan kualitas BAD
contoh_bad = df[df["Quality"] == "bad"].iloc[0]

print("CONTOH INPUT APEL KUALITAS BURUK")
print("--------------------------------")

fitur = [
    "Size", "Weight", "Sweetness", "Crunchiness",
    "Juiciness", "Ripeness", "Acidity"
]

for kolom in fitur:
    print(f"{kolom}: {contoh_bad[kolom]}")

print("--------------------------------")
print("Label asli:", contoh_bad["Quality"])