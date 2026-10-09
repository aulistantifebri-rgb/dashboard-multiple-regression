
from flask import Flask, render_template_string
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)

# 1. Membaca dataset
df = pd.read_csv("latihan1.csv")
df.columns = df.columns.str.strip()

# 2. Menentukan kolom target (Y)
if "Exam_Score" in df.columns:
    target = "Exam_Score"
else:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        raise ValueError("Dataset tidak memiliki kolom numerik.")
    target = numeric_cols[-1]

# 3. Menyiapkan data regresi
data = df.dropna(subset=[target]).copy()
y = pd.to_numeric(data[target], errors="coerce")
data = data.loc[y.notna()].copy()
y = y.loc[y.notna()]

X_raw = data.drop(columns=[target])
X = pd.get_dummies(X_raw, drop_first=True, dtype=float)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).fillna(0)

# 4. Membuat model Multiple Linear Regression
model = LinearRegression()
model.fit(X, y)

prediksi = model.predict(X)

r2 = r2_score(y, prediksi)
mae = mean_absolute_error(y, prediksi)
rmse = np.sqrt(mean_squared_error(y, prediksi))

koefisien = pd.DataFrame({
    "Variabel": X.columns,
    "Koefisien": model.coef_
})
koefisien["Koefisien"] = koefisien["Koefisien"].round(4)

# Menyiapkan data untuk grafik interaktif
data_grafik = df.astype(object).where(pd.notna(df), None)
kolom = df.columns.tolist()
data_json = data_grafik.to_dict(orient="records")

HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Dashboard Multiple Linear Regression</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f3f4f6;
            color: #222;
            margin: 0;
            padding: 24px;
        }
        h1 { color: #4338ca; }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }
        .card, .panel {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px #00000010;
            margin-bottom: 18px;
        }
        .number { font-size: 26px; font-weight: bold; color: #4f46e5; }
        select {
            padding: 10px;
            margin: 5px 10px 10px 0;
            border: 1px solid #ccc;
            border-radius: 7px;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; border-bottom: 1px solid #ddd; text-align: left; }
        th { background: #eef2ff; }
        .scroll { overflow-x: auto; }
        .note { color: #555; line-height: 1.6; }
    </style>
</head>
<body>
    <h1>Dashboard Multiple Linear Regression</h1>
    <p class="note">Analisis data menggunakan metode Multiple Linear Regression.</p>

    <div class="cards">
        <div class="card">
            <p>Jumlah Data</p>
            <div class="number">{{ jumlah }}</div>
        </div>
        <div class="card">
            <p>Jumlah Variabel X</p>
            <div class="number">{{ jumlah_x }}</div>
        </div>
        <div class="card">
            <p>R-squared (R²)</p>
            <div class="number">{{ r2 }}</div>
        </div>
        <div class="card">
            <p>RMSE</p>
            <div class="number">{{ rmse }}</div>
        </div>
    </div>

    <div class="panel">
        <h2>1. Grafik Interaktif X dan Y</h2>
        <p>Pilih variabel X dan Y untuk mengamati hubungan antarvariabel.</p>

        <label for="pilihX">Variabel X:</label>
        <select id="pilihX"></select>

        <label for="pilihY">Variabel Y:</label>
        <select id="pilihY"></select>

        <div id="grafik" style="width:100%;height:450px;"></div>
    </div>

    <div class="panel">
        <h2>2. Behind the Scenes: Proses Regresi</h2>
        <p><b>Target Y:</b> {{ target }}</p>
        <p><b>Metode:</b> Multiple Linear Regression</p>
        <p><b>Persamaan umum:</b> Y = a + b₁X₁ + b₂X₂ + ... + bnXn</p>
        <p class="note">
            Model mempelajari hubungan antara beberapa variabel X dan target Y.
            Data kategori diubah menjadi angka menggunakan one-hot encoding.
            Model kemudian menghitung koefisien regresi untuk menghasilkan prediksi.
        </p>
        <p><b>R²:</b> {{ r2 }} — mengukur kecocokan model pada data yang digunakan.</p>
        <p><b>MAE:</b> {{ mae }} — rata-rata nilai absolut kesalahan prediksi.</p>
        <p><b>RMSE:</b> {{ rmse }} — ukuran kesalahan prediksi yang memberi bobot lebih pada kesalahan besar.</p>
        <p><b>Catatan:</b> Evaluasi ini menggunakan data yang sama untuk melatih model, bukan data uji terpisah.</p>
    </div>

    <div class="panel">
        <h2>3. Koefisien Regresi</h2>
        <p>Koefisien menunjukkan perubahan prediksi saat satu variabel berubah,
        dengan variabel lainnya dianggap tetap.</p>
        <div class="scroll">
            <table>
                <tr><th>Variabel X</th><th>Koefisien</th></tr>
                {% for item in koefisien %}
                <tr>
                    <td>{{ item["Variabel"] }}</td>
                    <td>{{ item["Koefisien"] }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        <p><b>Intercept:</b> {{ intercept }}</p>
    </div>

    <script>
        const kolom = {{ kolom | tojson }};
        const baris = {{ data | tojson }};

        const pilihX = document.getElementById("pilihX");
        const pilihY = document.getElementById("pilihY");

        kolom.forEach(function(nama) {
            const opsiX = document.createElement("option");
            opsiX.value = nama;
            opsiX.textContent = nama;
            pilihX.appendChild(opsiX);

            const opsiY = document.createElement("option");
            opsiY.value = nama;
            opsiY.textContent = nama;
            pilihY.appendChild(opsiY);
        });

        pilihX.value = kolom[0];
        pilihY.value = kolom.includes("Exam_Score")
            ? "Exam_Score"
            : kolom[Math.min(1, kolom.length - 1)];

        function buatGrafik() {
            const xNama = pilihX.value;
            const yNama = pilihY.value;

            const xNilai = [];
            const yNilai = [];
            const label = [];

            baris.forEach(function(barisSatu) {
                const x = Number(barisSatu[xNama]);
                const y = Number(barisSatu[yNama]);

                if (
                    barisSatu[xNama] !== null &&
                    barisSatu[yNama] !== null &&
                    barisSatu[xNama] !== "" &&
                    barisSatu[yNama] !== "" &&
                    Number.isFinite(x) &&
                    Number.isFinite(y)
                ) {
                    xNilai.push(x);
                    yNilai.push(y);
                    label.push(String(barisSatu[xNama]) + ", " + String(barisSatu[yNama]));
                }
            });

            const angka = xNilai.length > 0;
            const jejak = [{
                x: xNilai,
                y: yNilai,
                text: label,
                type: "scatter",
                mode: "markers",
                marker: { size: 9 },
                name: "Data"
            }];

            if (!angka) {
                const kategoriX = [];
                const kategoriY = [];
                baris.forEach(function(r) {
                    if (r[xNama] !== null && r[yNama] !== null) {
                        kategoriX.push(String(r[xNama]));
                        kategoriY.push(String(r[yNama]));
                    }
                });
                jejak[0].x = kategoriX;
                jejak[0].y = kategoriY;
            }

            Plotly.newPlot("grafik", jejak, {
                title: xNama + " vs " + yNama,
                xaxis: { title: xNama },
                yaxis: { title: yNama },
                margin: { t: 55, l: 60, r: 25, b: 70 },
                paper_bgcolor: "white",
                plot_bgcolor: "white"
            }, { responsive: true });
        }

        pilihX.addEventListener("change", buatGrafik);
        pilihY.addEventListener("change", buatGrafik);
        buatGrafik();
    </script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    return render_template_string(
        HTML,
        jumlah=len(data),
        jumlah_x=len(X.columns),
        target=target,
        r2=round(r2, 4),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        intercept=round(float(model.intercept_), 4),
        koefisien=koefisien.to_dict(orient="records"),
        kolom=kolom,
        data=data_json
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)