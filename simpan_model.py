import joblib

# Memuat Model A
model = joblib.load("model_decision_tree_A.pkl")

# Menyimpan model terbaik menggunakan Joblib
joblib.dump(
    model,
    "model_terbaik.joblib"
)

print("Model terbaik berhasil disimpan!")
print("Nama file: model_terbaik.joblib")