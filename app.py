import streamlit as st
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="Motor Vehicle Analysis Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for center-aligned headers and layout
st.markdown("""
<style>
    /* Center align page title */
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 6px;
        color: inherit;
    }
    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 24px;
    }
    /* KPI cards */
    .kpi-card {
        background-color: #1e293b;
        padding: 20px 16px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #3b82f6;
        margin-bottom: 12px;
        text-align: center;
    }
    .kpi-card h3 {
        margin: 0 0 6px 0;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9ca3af;
    }
    .kpi-card p {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
        color: #f1f5f9;
    }
    /* Force center alignment on ALL dataframe table headers and cells */
    div[data-testid="stDataFrame"] th {
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] td {
        text-align: center !important;
    }
    /* Section headers */
    .section-label {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: inherit;
    }
    /* Metric label and value center alignment */
    div[data-testid="metric-container"] {
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    div[data-testid="metric-container"] > label {
        text-align: center;
        width: 100%;
    }
    div[data-testid="metric-container"] > div {
        text-align: center;
        width: 100%;
    }
    /* Confusion matrix centering */
    .confusion-wrapper {
        display: flex;
        justify-content: center;
    }
    hr { margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# Define paths
OUTPUT_DIR = r"D:\PROJEK BDA\output"
KPI_PATH = os.path.join(OUTPUT_DIR, "dashboard_kpi.csv")
BOROUGH_PATH = os.path.join(OUTPUT_DIR, "dashboard_borough_analysis.csv")
RISK_PATH = os.path.join(OUTPUT_DIR, "dashboard_risk_distribution.csv")
EVAL_PATH = os.path.join(OUTPUT_DIR, "model_evaluation.csv")
CONF_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
FORECAST_PATH = os.path.join(OUTPUT_DIR, "forecast_risk_next_month.csv")


# ==========================================
# CENTERED TITLE
# ==========================================
st.markdown('<h1 class="main-title">Motor Vehicle Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analisis dan Prediksi Kecelakaan Kendaraan Bermotor Menggunakan Hadoop HDFS dan Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")


# Helper function to safely load data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# Load datasets
df_kpi = load_data(KPI_PATH)
df_borough = load_data(BOROUGH_PATH)
df_risk = load_data(RISK_PATH)
df_eval = load_data(EVAL_PATH)
df_conf = load_data(CONF_PATH)
df_forecast = load_data(FORECAST_PATH)


# (Project Overview section removed - moved to bottom as Insight)


# ==========================================
# KPI METRICS
# ==========================================
st.markdown('<div class="section-label">Ringkasan KPI Kecelakaan</div>', unsafe_allow_html=True)

if not df_kpi.empty:
    kpi_dict = dict(zip(df_kpi['metric'], df_kpi['value']))

    total_accidents = int(kpi_dict.get('total_data_kecelakaan', 0))
    total_injured = int(kpi_dict.get('total_injured', 0))
    total_killed = int(kpi_dict.get('total_killed', 0))
    total_boroughs = int(kpi_dict.get('total_borough', 0))

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f'<div class="kpi-card"><h3>Total Kecelakaan</h3><p>{total_accidents:,}</p></div>', unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(f'<div class="kpi-card"><h3>Korban Luka</h3><p>{total_injured:,}</p></div>', unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(f'<div class="kpi-card"><h3>Korban Meninggal</h3><p>{total_killed:,}</p></div>', unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(f'<div class="kpi-card"><h3>Wilayah (Borough)</h3><p>{total_boroughs:,}</p></div>', unsafe_allow_html=True)
else:
    st.warning("Data KPI tidak tersedia.")

st.markdown("---")


# ==========================================
# BOROUGH ANALYSIS & RISK DISTRIBUTION (SIDE BY SIDE)
# ==========================================
analysis_cols = st.columns(2)

with analysis_cols[0]:
    st.markdown('<div class="section-label">Borough Analysis (Hadoop MapReduce)</div>', unsafe_allow_html=True)
    st.caption("Jumlah kecelakaan per wilayah administratif hasil agregasi MapReduce.")

    if not df_borough.empty:
        st.bar_chart(
            data=df_borough.set_index('borough')['total_crash'],
            color="#2563eb"
        )
        df_borough_display = df_borough.copy()
        df_borough_display.columns = [c.replace("_", " ").title() for c in df_borough_display.columns]
        st.dataframe(
            df_borough_display.style
                .format({
                    "Total Crash": "{:,.0f}",
                    "Total Injured": "{:,.0f}",
                    "Total Killed": "{:,.0f}"
                })
                .set_properties(**{"text-align": "center"})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
            use_container_width=True
        )
    else:
        st.warning("Data Borough tidak tersedia.")

with analysis_cols[1]:
    st.markdown('<div class="section-label">Distribusi Tingkat Risiko (Random Forest Classifier)</div>', unsafe_allow_html=True)
    st.caption("Distribusi tingkat risiko kecelakaan berdasarkan jumlah korban pada data historis.")

    if not df_risk.empty:
        st.bar_chart(
            data=df_risk.set_index('risk_level')['total'],
            color="#d97706"
        )
        df_risk_display = df_risk.copy()
        df_risk_display.columns = [c.replace("_", " ").title() for c in df_risk_display.columns]
        st.dataframe(
            df_risk_display.style
                .format({"Total": "{:,.0f}"})
                .set_properties(**{"text-align": "center"})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
            use_container_width=True
        )
    else:
        st.warning("Data tingkat risiko tidak tersedia.")





# ==========================================
# PREDICTIVE FORECASTING (1 MONTH AHEAD)
# ==========================================
st.markdown('<div class="section-label">Peramalan Risiko Kecelakaan (1 Bulan Ke Depan)</div>', unsafe_allow_html=True)
st.caption("Hasil prediksi peramalan jumlah kecelakaan harian beserta tingkat risikonya untuk 30 hari ke depan dengan faktor eksternal volume lalu lintas akhir pekan.")

if not df_forecast.empty:
    st.line_chart(
        data=df_forecast.set_index('date')['predicted_crash_count'],
        color="#dc2626"
    )

    st.markdown("**Detail Tabel Peramalan**")

    def style_risk_level(val):
        if val == 'High':
            return 'background-color: #fee2e2; color: #991b1b; font-weight: bold;'
        elif val == 'Medium':
            return 'background-color: #fef3c7; color: #92400e; font-weight: bold;'
        else:
            return 'background-color: #dcfce7; color: #166534; font-weight: bold;'

    df_forecast_display = df_forecast.copy()
    df_forecast_display.columns = [c.replace("_", " ").title() for c in df_forecast_display.columns]

    st.dataframe(
        df_forecast_display.style
            .applymap(style_risk_level, subset=['Risk Level'])
            .format({"Predicted Crash Count": "{:,.0f}"})
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
        use_container_width=True
    )
else:
    st.warning("Data peramalan 30 hari ke depan tidak tersedia.")


# ==========================================
# INSIGHT SECTION (DATA-DRIVEN, BOTTOM)
# ==========================================
st.markdown("---")
st.markdown('<div class="section-label">Insight Analisis Data</div>', unsafe_allow_html=True)
st.caption("Kesimpulan dan temuan utama berdasarkan seluruh data yang ditampilkan pada dashboard ini.")

if not df_kpi.empty and not df_borough.empty and not df_risk.empty and not df_eval.empty and not df_forecast.empty:
    kpi_d = dict(zip(df_kpi['metric'], df_kpi['value']))
    total_kec = int(kpi_d.get('total_data_kecelakaan', 0))
    total_luka = int(kpi_d.get('total_injured', 0))
    total_mati = int(kpi_d.get('total_killed', 0))
    high_risk = int(kpi_d.get('total_high_risk', 0))
    medium_risk = int(kpi_d.get('total_medium_risk', 0))
    low_risk = int(kpi_d.get('total_low_risk', 0))

    top_borough = df_borough.iloc[0]
    second_borough = df_borough.iloc[1]

    high_pct = high_risk / total_kec * 100
    medium_pct = medium_risk / total_kec * 100
    low_pct = low_risk / total_kec * 100

    eval_d = dict(zip(df_eval['metric'], df_eval['score']))
    acc = eval_d.get('accuracy', 0) * 100
    prec = eval_d.get('precision', 0) * 100
    recall = eval_d.get('recall', 0) * 100
    f1 = eval_d.get('f1_score', 0) * 100

    n_high_days = len(df_forecast[df_forecast['risk_level'] == 'High'])
    n_medium_days = len(df_forecast[df_forecast['risk_level'] == 'Medium'])
    n_low_days = len(df_forecast[df_forecast['risk_level'] == 'Low'])
    max_forecast = df_forecast.loc[df_forecast['predicted_crash_count'].idxmax()]
    min_forecast = df_forecast.loc[df_forecast['predicted_crash_count'].idxmin()]

    ins_col1, ins_col2 = st.columns(2)

    with ins_col1:
        st.markdown("**Volume & Sebaran Kecelakaan (KPI)**")
        st.info(
            f"Dataset mencakup **{total_kec:,} kecelakaan** dari tahun 2012 hingga 2024, "
            f"menghasilkan **{total_luka:,} korban luka** dan **{total_mati:,} korban meninggal**. "
            f"Ini menunjukkan bahwa rata-rata setiap kecelakaan mengakibatkan "
            f"**{total_luka/total_kec:.2f} korban luka** dan setiap **{total_kec//total_mati:,} kecelakaan** "
            f"menyebabkan satu kematian."
        )

        st.markdown("**Borough Analysis (Hadoop MapReduce)**")
        st.warning(
            f"Wilayah **{top_borough['borough']}** mencatat volume kecelakaan tertinggi "
            f"(**{int(top_borough['total_crash']):,} kasus**), diikuti oleh **{second_borough['borough']}** "
            f"(**{int(second_borough['total_crash']):,} kasus**). "
            f"Hal ini menunjukkan bahwa wilayah padat aktivitas dan populasi memiliki volume insiden "
            f"yang jauh lebih tinggi dibandingkan wilayah pinggiran seperti Staten Island "
            f"({int(df_borough.iloc[-1]['total_crash']):,} kasus)."
        )

    with ins_col2:
        st.markdown("**Distribusi Tingkat Risiko (Klasifikasi ML)**")
        st.success(
            f"Mayoritas kecelakaan ({low_pct:.1f}%) dikategorikan **Low Risk** ({low_risk:,} kejadian), "
            f"artinya tidak melibatkan korban luka ataupun meninggal. Namun, sebesar **{high_pct:.1f}%** "
            f"({high_risk:,} kejadian) tergolong **High Risk** (korban luka >= 2 atau ada yang meninggal), "
            f"dan **{medium_pct:.1f}%** ({medium_risk:,} kejadian) **Medium Risk**. "
            f"Ini menegaskan pentingnya intervensi preventif terhadap sekitar **{high_risk + medium_risk:,} "
            f"insiden berisiko tinggi dan sedang**."
        )

        st.markdown("**Evaluasi Model Random Forest Classifier**")
        st.info(
            f"Model Random Forest mencapai akurasi **{acc:.2f}%** dengan presisi **{prec:.2f}%**, "
            f"recall **{recall:.2f}%**, dan F1-Score **{f1:.2f}%**. "
            f"Model cenderung lebih akurat dalam mengidentifikasi kelas Low Risk (kelas mayoritas) "
            f"dibandingkan High Risk (kelas minoritas). Peningkatan performa dapat dicapai dengan "
            f"teknik oversampling (SMOTE) atau penambahan fitur eksternal."
        )

    st.markdown("**Peramalan Risiko 30 Hari Ke Depan (Time-Series Forecasting)**")
    st.error(
        f"Dari 30 hari yang diprediksi, terdapat **{n_high_days} hari High Risk**, "
        f"**{n_medium_days} hari Medium Risk**, dan **{n_low_days} hari Low Risk**. "
        f"Hari dengan risiko tertinggi adalah **{max_forecast['day_of_week']} ({max_forecast['date']})** "
        f"dengan prediksi **{int(max_forecast['predicted_crash_count']):,} kecelakaan** — "
        f"dipicu oleh faktor eksternal lonjakan traffic akhir pekan (Saturday multiplier 2.2x). "
        f"Sebaliknya, hari paling aman adalah **{min_forecast['day_of_week']} ({min_forecast['date']})** "
        f"dengan prediksi hanya **{int(min_forecast['predicted_crash_count']):,} kecelakaan**. "
        f"Otoritas setempat disarankan meningkatkan patroli dan pengawasan lalu lintas pada akhir pekan."
    )
else:
    st.warning("Data tidak cukup untuk menampilkan insight.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.85rem;'>Motor Vehicle Collisions Big Data Analytics Project</p>", unsafe_allow_html=True)
