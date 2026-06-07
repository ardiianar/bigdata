"""
Risk Prediction from Hadoop - Random Forest Classifier
Pipeline: Preprocessing → MapReduce Integration → Feature Engineering → Training → Evaluasi → Dashboard
"""

# Import library yang dibutuhkan
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


# Definisi path file input dan output
DATA_PATH = r"D:\PROJEK BDA\output\motor_vehicle_clean.csv"
BOROUGH_PATH = r"D:\PROJEK BDA\output\borough_analysis.txt"

OUTPUT_PREDICTION        = r"D:\PROJEK BDA\output\risk_prediction_from_hadoop.csv"
OUTPUT_EVALUATION        = r"D:\PROJEK BDA\output\model_evaluation.csv"
OUTPUT_CONFUSION         = r"D:\PROJEK BDA\output\confusion_matrix.csv"

OUTPUT_DASHBOARD_KPI     = r"D:\PROJEK BDA\output\dashboard_kpi.csv"
OUTPUT_DASHBOARD_BOROUGH = r"D:\PROJEK BDA\output\dashboard_borough_analysis.csv"
OUTPUT_DASHBOARD_RISK    = r"D:\PROJEK BDA\output\dashboard_risk_distribution.csv"
OUTPUT_DASHBOARD_FACTOR  = r"D:\PROJEK BDA\output\dashboard_factor_causes.csv"


# Membaca dataset bersih hasil preprocessing
print("Membaca dataset clean...")
df = pd.read_csv(DATA_PATH)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(".", "", regex=False)
)

print("Jumlah data:", len(df))
print("Kolom dataset:")
print(df.columns.tolist())


# Membaca hasil Borough Analysis dari Hadoop MapReduce
print("\nMembaca hasil MapReduce borough_analysis.txt...")

borough_counts = {}

with open(BOROUGH_PATH, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) >= 2:
            total = int(parts[-1])
            borough = " ".join(parts[:-1])
            borough_counts[borough] = total

borough_df = pd.DataFrame(
    list(borough_counts.items()),
    columns=["borough", "borough_total_crash"]
)

print("\nHasil MapReduce:")
print(borough_df)


# Menggabungkan (merge/join) dataset dengan hasil MapReduce berdasarkan kolom borough
if "borough" not in df.columns:
    raise ValueError("Kolom borough tidak ditemukan di dataset.")

df["borough"] = df["borough"].fillna("Unknown")
df["borough"] = df["borough"].astype(str).str.strip()
df["borough"] = df["borough"].replace("", "Unknown")

print("\nMenggabungkan dataset dengan hasil MapReduce...")

df = df.merge(borough_df, on="borough", how="left")
df["borough_total_crash"] = df["borough_total_crash"].fillna(0)

print(df[["borough", "borough_total_crash"]].head())


# Mencari kolom korban luka dan korban meninggal secara dinamis
print("\nMengecek kolom korban kecelakaan...")
print("Daftar kolom yang tersedia:")
print(df.columns.tolist())


def find_column(possible_names):
    """Mencari nama kolom dari daftar kemungkinan nama."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


injured_col = find_column([
    "number_of_persons_injured",
    "number_persons_injured",
    "persons_injured",
    "person_injured",
    "injured",
    "total_injured",
    "jumlah_korban_luka"
])

killed_col = find_column([
    "number_of_persons_killed",
    "number_persons_killed",
    "persons_killed",
    "person_killed",
    "killed",
    "total_killed",
    "jumlah_korban_meninggal"
])

if injured_col is None:
    print("\nKolom korban luka tidak ditemukan.")
    print("Kolom yang tersedia:")
    print(df.columns.tolist())
    raise ValueError("Tidak ada kolom untuk jumlah korban luka.")

if killed_col is None:
    print("\nKolom korban meninggal tidak ditemukan.")
    print("Kolom yang tersedia:")
    print(df.columns.tolist())
    raise ValueError("Tidak ada kolom untuk jumlah korban meninggal.")

print(f"Kolom korban luka yang digunakan: {injured_col}")
print(f"Kolom korban meninggal yang digunakan: {killed_col}")

df["number_of_persons_injured"] = pd.to_numeric(
    df[injured_col], errors="coerce"
).fillna(0)

df["number_of_persons_killed"] = pd.to_numeric(
    df[killed_col], errors="coerce"
).fillna(0)


# Membuat label risiko (High / Medium / Low) berdasarkan jumlah korban
def create_risk(row):
    injured = row["number_of_persons_injured"]
    killed  = row["number_of_persons_killed"]
    if killed > 0 or injured >= 2:
        return "High"
    elif injured == 1:
        return "Medium"
    else:
        return "Low"


df["risk_level"] = df.apply(create_risk, axis=1)

print("\nDistribusi risk_level:")
print(df["risk_level"].value_counts())


# Feature engineering: ekstraksi fitur tanggal dan waktu
if "crash_date" in df.columns:
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    df["crash_year"]      = df["crash_date"].dt.year
    df["crash_month"]     = df["crash_date"].dt.month
    df["crash_day"]       = df["crash_date"].dt.day
    df["crash_dayofweek"] = df["crash_date"].dt.dayofweek
else:
    df["crash_year"] = df["crash_month"] = df["crash_day"] = df["crash_dayofweek"] = 0

# Feature engineering: fitur biner kondisi waktu (weekend, malam, jam sibuk)
if "hour" in df.columns:
    df["is_weekend"]   = (df["crash_dayofweek"] >= 5).astype(int)
    df["is_night"]     = ((df["hour"] >= 22) | (df["hour"] <= 4)).astype(int)
    df["is_rush_hour"] = (((df["hour"] >= 7) & (df["hour"] <= 9)) |
                          ((df["hour"] >= 16) & (df["hour"] <= 19))).astype(int)
else:
    df["is_weekend"] = df["is_night"] = df["is_rush_hour"] = 0

# Feature engineering: koordinat spasial (latitude, longitude)
for col in ["latitude", "longitude"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            df[col].median() if col in df.columns else 0
        )


# Target encoding: proporsi historis risiko per faktor penyebab dan jenis kendaraan
for col in ["contributing_factor", "vehicle_type"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str)
        high_rate = df.groupby(col)["risk_level"].apply(
            lambda s: (s == "High").mean()
        ).rename(f"{col}_high_rate")
        med_rate = df.groupby(col)["risk_level"].apply(
            lambda s: (s == "Medium").mean()
        ).rename(f"{col}_medium_rate")
        df = df.merge(high_rate, on=col, how="left")
        df = df.merge(med_rate,  on=col, how="left")


# Feature engineering: statistik rata-rata korban per borough, faktor, kendaraan, dan jam
print("\nMembuat fitur aggregate injury statistics...")

for col in ["contributing_factor", "vehicle_type", "borough"]:
    if col in df.columns:
        avg_inj = (
            df.groupby(col)["number_of_persons_injured"]
            .mean()
            .rename(f"{col}_avg_injured")
        )
        avg_kill = (
            df.groupby(col)["number_of_persons_killed"]
            .mean()
            .rename(f"{col}_avg_killed")
        )
        df = df.merge(avg_inj,  on=col, how="left")
        df = df.merge(avg_kill, on=col, how="left")

if "hour" in df.columns:
    avg_inj_hour = (
        df.groupby("hour")["number_of_persons_injured"]
        .mean()
        .rename("hour_avg_injured")
    )
    avg_kill_hour = (
        df.groupby("hour")["number_of_persons_killed"]
        .mean()
        .rename("hour_avg_killed")
    )
    df = df.merge(avg_inj_hour,  on="hour", how="left")
    df = df.merge(avg_kill_hour, on="hour", how="left")


# Memilih fitur input model (feature selection)
candidate_numeric = [
    "borough_total_crash",
    "crash_year", "crash_month", "crash_day", "crash_dayofweek",
    "hour",
    "is_weekend", "is_night", "is_rush_hour",
    "latitude", "longitude",
    "contributing_factor_high_rate", "contributing_factor_medium_rate",
    "vehicle_type_high_rate", "vehicle_type_medium_rate",
    "contributing_factor_avg_injured", "contributing_factor_avg_killed",
    "vehicle_type_avg_injured", "vehicle_type_avg_killed",
    "borough_avg_injured", "borough_avg_killed",
    "hour_avg_injured", "hour_avg_killed",
]

candidate_cat = ["borough"]

num_feats = [c for c in candidate_numeric if c in df.columns]
cat_feats = [c for c in candidate_cat if c in df.columns]
features  = cat_feats + num_feats

print("\nFitur yang digunakan:")
print(features)

X = df[features].copy()
y = df["risk_level"]

for col in cat_feats:
    X[col] = X[col].fillna("Unknown").astype(str)
for col in num_feats:
    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)


# Membagi data menjadi training set (80%) dan testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Membangun pipeline: preprocessing (OneHotEncoder) + model Random Forest Classifier
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ("num", "passthrough", num_feats)
    ]
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Melatih model menggunakan data training
print("\nTraining Random Forest...")
pipeline.fit(X_train, y_train)


# Mengevaluasi performa model: accuracy, precision, recall, F1, confusion matrix
print("\nEvaluasi model...")
y_pred = pipeline.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nHasil Evaluasi:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

evaluation_df = pd.DataFrame({
    "metric": ["accuracy", "precision", "recall", "f1_score"],
    "score":  [accuracy, precision, recall, f1]
})

confusion_df = pd.DataFrame(
    cm,
    index=[f"actual_{label}" for label in labels],
    columns=[f"predicted_{label}" for label in labels]
)

# Menyimpan hasil evaluasi model ke file CSV
evaluation_df.to_csv(OUTPUT_EVALUATION, index=False)
confusion_df.to_csv(OUTPUT_CONFUSION)


# Memprediksi risk level untuk seluruh dataset beserta nilai probabilitasnya
print("\nMembuat prediksi untuk semua data...")
df["predicted_risk"] = pipeline.predict(X)

prediction_proba = pipeline.predict_proba(X)
class_names = pipeline.classes_

for idx, class_name in enumerate(class_names):
    df[f"probability_{class_name}"] = prediction_proba[:, idx]


# Menyimpan hasil prediksi ke file CSV
selected_columns = [
    col for col in [
        "crash_date",
        "crash_time",
        "borough",
        "borough_total_crash",
        "number_of_persons_injured",
        "number_of_persons_killed",
        "contributing_factor_vehicle_1",
        "vehicle_type_code1",
        "risk_level",
        "predicted_risk",
        "probability_High",
        "probability_Low",
        "probability_Medium"
    ]
    if col in df.columns
]

df[selected_columns].to_csv(OUTPUT_PREDICTION, index=False)

print("\nFile risk prediction berhasil dibuat:")
print(OUTPUT_PREDICTION)


# Membuat file-file CSV untuk keperluan dashboard Streamlit
print("\nMembuat file dashboard...")

# KPI: total kecelakaan, korban, dan distribusi risiko
dashboard_kpi = pd.DataFrame({
    "metric": [
        "total_data_kecelakaan",
        "total_injured",
        "total_killed",
        "total_borough",
        "total_high_risk",
        "total_medium_risk",
        "total_low_risk"
    ],
    "value": [
        len(df),
        df["number_of_persons_injured"].sum(),
        df["number_of_persons_killed"].sum(),
        df["borough"].nunique(),
        (df["predicted_risk"] == "High").sum(),
        (df["predicted_risk"] == "Medium").sum(),
        (df["predicted_risk"] == "Low").sum()
    ]
})
dashboard_kpi.to_csv(OUTPUT_DASHBOARD_KPI, index=False)

# Borough analysis: agregasi total kecelakaan per wilayah
dashboard_borough = (
    df.groupby("borough")
    .agg(
        total_crash=("borough", "count"),
        total_injured=("number_of_persons_injured", "sum"),
        total_killed=("number_of_persons_killed", "sum")
    )
    .reset_index()
    .sort_values("total_crash", ascending=False)
)
dashboard_borough.to_csv(OUTPUT_DASHBOARD_BOROUGH, index=False)

# Distribusi risk level hasil prediksi model
dashboard_risk = (
    df["predicted_risk"]
    .value_counts()
    .reset_index()
)
dashboard_risk.columns = ["risk_level", "total"]
dashboard_risk.to_csv(OUTPUT_DASHBOARD_RISK, index=False)

# Top 20 faktor penyebab kecelakaan terbanyak
if "contributing_factor" in df.columns:
    dashboard_factor = (
        df["contributing_factor"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .reset_index()
        .head(20)
    )
    dashboard_factor.columns = ["factor", "total"]
    dashboard_factor.to_csv(OUTPUT_DASHBOARD_FACTOR, index=False)

print("\nSelesai.")
print("Semua file hasil ML dan dashboard tersimpan di D:\\PROJEK BDA\\output")