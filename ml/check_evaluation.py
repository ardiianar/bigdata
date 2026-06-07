"""
Script untuk menampilkan hasil evaluasi model analisis risiko dan prediksi risiko 1 bulan ke depan.
"""

import pandas as pd
import os

# PROSES 1: MENENTUKAN PATH FILE SUMBER DATA
EVAL_PATH = r"D:\PROJEK BDA\output\model_evaluation.csv"
CONF_PATH = r"D:\PROJEK BDA\output\confusion_matrix.csv"
FORECAST_PATH = r"D:\PROJEK BDA\output\forecast_risk_next_month.csv"

# PROSES 2: MEMBACA DAN MENAMPILKAN METRIK EVALUASI CLASSIFICATION MODEL
print("=" * 60)
print("     HASIL EVALUASI MODEL RANDOM FOREST CLASSIFIER")
print("=" * 60)

if os.path.exists(EVAL_PATH):
    df_eval = pd.read_csv(EVAL_PATH)
    print("\n[ METRIK EVALUASI ]\n")
    for _, row in df_eval.iterrows():
        label = row["metric"].replace("_", " ").upper()
        score = float(row["score"]) * 100
        bar   = "#" * int(score // 5)
        print(f"  {label:<12} : {score:6.2f}%  {bar}")
else:
    print(f"\n[ERROR] File tidak ditemukan: {EVAL_PATH}")
    print("Jalankan dulu: python ml/risk_prediction_from_hadoop.py")

# PROSES 3: MEMBACA DAN MENAMPILKAN CONFUSION MATRIX
print("\n[ CONFUSION MATRIX ]\n")
if os.path.exists(CONF_PATH):
    df_conf = pd.read_csv(CONF_PATH, index_col=0)
    df_conf.columns = [c.replace("predicted_", "pred_") for c in df_conf.columns]
    df_conf.index   = [i.replace("actual_", "act_") for i in df_conf.index]
    print(df_conf.to_string())
    print()
    print("  Diagonal = prediksi BENAR  |  Luar diagonal = prediksi SALAH")
else:
    print(f"[ERROR] File tidak ditemukan: {CONF_PATH}")

# PROSES 4: MEMBACA DAN MENAMPILKAN HASIL PREDIKSI RISIKO 1 BULAN KE DEPAN
print("\n" + "=" * 60)
print("     HASIL PREDIKSI RISIKO 1 BULAN KE DEPAN (30 HARI)")
print("=" * 60)

if os.path.exists(FORECAST_PATH):
    df_forecast = pd.read_csv(FORECAST_PATH)
    print("\n[ TREN PREDIKSI RISIKO HARIAN ]\n")
    print(df_forecast.to_string(index=False))
    
    # Menampilkan ringkasan distribusi risiko
    print("\n[ RINGKASAN DISTRIBUSI TINGKAT RISIKO ]\n")
    risk_counts = df_forecast["risk_level"].value_counts()
    for level, count in risk_counts.items():
        print(f"  Tingkat Risiko {level:<7} : {count:2d} Hari")
else:
    print(f"\n[ERROR] File tidak ditemukan: {FORECAST_PATH}")
    print("Jalankan dulu: python ml/forecast_risk_next_month.py")

print("\n" + "=" * 60)
print("  Lokasi File Output:")
print(f"    - {EVAL_PATH}")
print(f"    - {CONF_PATH}")
print(f"    - {FORECAST_PATH}")
print("=" * 60)

