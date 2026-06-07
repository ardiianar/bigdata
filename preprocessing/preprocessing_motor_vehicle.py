# Mengimpor library pandas untuk membaca dan membersihkan dataset
import pandas as pd

# Menentukan lokasi dataset mentah
input_path = "d:/PROJEK BDA/data/raw/Motor_Vehicle_Collisions_-_Crashes.csv"

# Menentukan lokasi hasil preprocessing
output_path = "d:/PROJEK BDA/data/processed/motor_vehicle_clean.csv"
output_path_backup = "d:/PROJEK BDA/output/motor_vehicle_clean.csv"

# Membaca dataset CSV
df = pd.read_csv(input_path)

# Menampilkan jumlah data dan kolom awal
print("Jumlah data awal:", df.shape[0])
print("Jumlah kolom awal:", df.shape[1])

# Memilih kolom yang digunakan untuk analisis
df = df[
    [
        "CRASH DATE",
        "CRASH TIME",
        "BOROUGH",
        "LATITUDE",
        "LONGITUDE",
        "CONTRIBUTING FACTOR VEHICLE 1",
        "VEHICLE TYPE CODE 1",
        "NUMBER OF PERSONS INJURED",
        "NUMBER OF PERSONS KILLED"
    ]
]

# Mengubah nama kolom agar lebih mudah diproses
df = df.rename(
    columns={
        "CRASH DATE": "crash_date",
        "CRASH TIME": "crash_time",
        "BOROUGH": "borough",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "CONTRIBUTING FACTOR VEHICLE 1": "contributing_factor",
        "VEHICLE TYPE CODE 1": "vehicle_type",
        "NUMBER OF PERSONS INJURED": "number_of_persons_injured",
        "NUMBER OF PERSONS KILLED": "number_of_persons_killed"
    }
)

# Menghapus data duplikat
df = df.drop_duplicates()

# Menghapus data kosong pada kolom penting
df = df.dropna(
    subset=[
        "crash_date",
        "crash_time",
        "contributing_factor",
        "vehicle_type"
    ]
)

# Mengisi nilai kosong pada borough
df["borough"] = df["borough"].fillna("Unknown")

# Menghapus data yang tidak memiliki koordinat
df = df.dropna(subset=["latitude", "longitude"])

# Mengubah kolom tanggal menjadi format datetime
df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")

# Mengubah kolom waktu menjadi format datetime
df["crash_time"] = pd.to_datetime(df["crash_time"], format="%H:%M", errors="coerce")

# Membuat fitur tahun
df["year"] = df["crash_date"].dt.year

# Membuat fitur bulan
df["month"] = df["crash_date"].dt.month

# Membuat fitur hari
df["day"] = df["crash_date"].dt.day

# Membuat fitur hari dalam minggu
df["day_of_week"] = df["crash_date"].dt.dayofweek + 1

# Membuat fitur jam
df["hour"] = df["crash_time"].dt.hour

# Menghapus data yang gagal dikonversi tanggal atau waktunya
df = df.dropna(subset=["year", "month", "day", "day_of_week", "hour"])

# Mengubah tipe data fitur waktu menjadi integer
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)
df["day"] = df["day"].astype(int)
df["day_of_week"] = df["day_of_week"].astype(int)
df["hour"] = df["hour"].astype(int)

# Membuat data agregasi risiko berdasarkan borough, bulan, hari, dan jam
risk_df = df.groupby(
    ["borough", "month", "day_of_week", "hour"]
).size().reset_index(name="crash_count")

# Menghitung rata-rata jumlah kecelakaan
avg_crash = risk_df["crash_count"].mean()

# Membuat label risiko kecelakaan
risk_df["risk_level"] = risk_df["crash_count"].apply(
    lambda x: "High Risk" if x >= avg_crash else "Low Risk"
)

# Menyimpan data bersih hasil preprocessing
df.to_csv(output_path, index=False)
df.to_csv(output_path_backup, index=False)

# Menyimpan data risk prediction
risk_df.to_csv(
    "d:/PROJEK BDA/output/risk_prediction.csv",
    index=False
)

# Menampilkan jumlah data setelah preprocessing
print("Jumlah data setelah preprocessing:", df.shape[0])

# Menampilkan jumlah data risk prediction
print("Jumlah data risk prediction:", risk_df.shape[0])

# Menampilkan pesan selesai
print("Preprocessing selesai.")