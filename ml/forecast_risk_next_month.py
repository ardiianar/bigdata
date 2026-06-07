import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

# =========================
# PATH FILE
# =========================
DATA_PATH = r"D:\PROJEK BDA\data\processed\motor_vehicle_clean.csv"
OUTPUT_PATH = r"D:\PROJEK BDA\output\forecast_risk_next_month.csv"
WEBSITE_OUTPUT_PATH = r"D:\PROJEK BDA\website\data\forecast_risk_next_month.csv"

print("Membaca dataset untuk peramalan...")
df = pd.read_csv(DATA_PATH)
df["crash_date"] = pd.to_datetime(df["crash_date"])

# Mengelompokkan data berdasarkan tanggal untuk menghitung jumlah kecelakaan harian
daily_crashes = df.groupby("crash_date").size().reset_index(name="crash_count")
daily_crashes = daily_crashes.sort_values("crash_date").reset_index(drop=True)

print(f"Total data harian: {len(daily_crashes)} hari")
print(f"Rentang tanggal: {daily_crashes['crash_date'].min().strftime('%Y-%m-%d')} s/d {daily_crashes['crash_date'].max().strftime('%Y-%m-%d')}")

# =========================
# MEMBUAT FITUR TIME SERIES
# =========================
def create_features(data):
    df_feat = data.copy()
    # Lag features
    df_feat["lag_1"] = df_feat["crash_count"].shift(1)
    df_feat["lag_2"] = df_feat["crash_count"].shift(2)
    df_feat["lag_7"] = df_feat["crash_count"].shift(7)
    df_feat["lag_14"] = df_feat["crash_count"].shift(14)
    
    # Rolling mean features
    df_feat["rolling_mean_7"] = df_feat["crash_count"].shift(1).rolling(window=7).mean()
    df_feat["rolling_mean_30"] = df_feat["crash_count"].shift(1).rolling(window=30).mean()
    
    # Date parts
    df_feat["dayofweek"] = df_feat["crash_date"].dt.dayofweek
    df_feat["day"] = df_feat["crash_date"].dt.day
    df_feat["month"] = df_feat["crash_date"].dt.month
    
    return df_feat.dropna().reset_index(drop=True)

df_features = create_features(daily_crashes)

X = df_features[["lag_1", "lag_2", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30", "dayofweek", "day", "month"]]
y = df_features["crash_count"]

# =========================
# TRAINING MODEL REGRESOR
# =========================
print("Melatih model Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# =========================
# PERAMALAN 30 HARI KE DEPAN
# =========================
print("Melakukan peramalan untuk 30 hari ke depan...")
last_date = daily_crashes["crash_date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Kita akan memprediksi satu per satu secara rekursif karena lag feed back
history = daily_crashes.copy()

for future_date in future_dates:
    # Buat baris baru untuk tanggal masa depan
    new_row = pd.DataFrame({"crash_date": [future_date], "crash_count": [0]})
    history = pd.concat([history, new_row], ignore_index=True)
    
    # Hitung fitur untuk baris terakhir
    temp_features = create_features(history)
    last_feat_row = temp_features.tail(1)[["lag_1", "lag_2", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30", "dayofweek", "day", "month"]]
    
    # Prediksi jumlah kecelakaan
    pred_count = model.predict(last_feat_row)[0]
    
    # Masukkan hasil prediksi kembali ke history
    history.loc[history.index[-1], "crash_count"] = pred_count

# Ambil hasil prediksi 30 hari terakhir
forecast_result = history.tail(30).copy()

# =========================
# FAKTOR EKSTERNAL: TRAFFIC WEEKEND & DAY OF WEEK MULTIPLIERS
# =========================
# Menambahkan faktor eksternal volume lalu lintas. Hari kerja memiliki baseline multiplier (1.0).
# Jumat s/d Minggu (weekend) memiliki peningkatan aktivitas lalu lintas sehingga peluang kecelakaan lebih tinggi.
def get_traffic_multiplier(date):
    dayofweek = date.dayofweek
    if dayofweek == 4:    # Friday
        return 1.5
    elif dayofweek == 5:  # Saturday
        return 2.2
    elif dayofweek == 6:  # Sunday
        return 1.8
    else:                 # Monday - Thursday
        return 1.0

forecast_result["traffic_multiplier"] = forecast_result["crash_date"].apply(get_traffic_multiplier)
forecast_result["predicted_crash_count"] = (forecast_result["crash_count"] * forecast_result["traffic_multiplier"]).round().astype(int)

# =========================
# KLASIFIKASI TINGKAT RISIKO DENGAN FAKTOR EKSTERNAL
# =========================
# Low: <= 300, Medium: 301 - 450, High: > 450
def get_risk_level(count):
    if count <= 300:
        return "Low"
    elif count <= 450:
        return "Medium"
    else:
        return "High"

forecast_result["risk_level"] = forecast_result["predicted_crash_count"].apply(get_risk_level)
forecast_result["date"] = forecast_result["crash_date"].dt.strftime("%Y-%m-%d")

# Nama hari dalam Bahasa Indonesia/Inggris
days_mapping = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday"
}
forecast_result["day_of_week"] = forecast_result["crash_date"].dt.dayofweek.map(days_mapping)

# Pilih kolom untuk output
output_df = forecast_result[["date", "day_of_week", "predicted_crash_count", "risk_level"]]

# Simpan ke folder output dan website
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(WEBSITE_OUTPUT_PATH), exist_ok=True)

output_df.to_csv(OUTPUT_PATH, index=False)
output_df.to_csv(WEBSITE_OUTPUT_PATH, index=False)

print("\nHasil Peramalan 1 Bulan Ke Depan:")
print(output_df.head(10))

print(f"\nHasil peramalan berhasil disimpan ke:")
print(f"- {OUTPUT_PATH}")
print(f"- {WEBSITE_OUTPUT_PATH}")
