from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Memuat model terbaik
model = joblib.load("model_terbaik.joblib")

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


@app.route("/", methods=["GET", "POST"])
def index():

    hasil = None
    error = None
    data_input = {}

    if request.method == "POST":

        try:
            # Mengambil input pengguna
            for nama_fitur in fitur:
                data_input[nama_fitur] = float(
                    request.form[nama_fitur]
                )

            # Mengubah input menjadi DataFrame
            data = pd.DataFrame(
                [data_input],
                columns=fitur
            )

            # Melakukan prediksi
            prediksi = model.predict(data)[0]

            # Menentukan label kualitas
            if prediksi == 1:
                hasil = "GOOD"
            else:
                hasil = "BAD"

        except ValueError:
            error = "Pastikan semua nilai yang dimasukkan berupa angka."

        except Exception as e:
            error = f"Terjadi kesalahan: {e}"

    return render_template(
        "index.html",
        hasil=hasil,
        error=error,
        data_input=data_input
    )


if __name__ == "__main__":
    app.run(debug=True)