#  Motor Vehicle Collision Big Data Analytics & Forecasting Dashboard

# Motor Vehicle Collision Big Data Analytics & Risk Prediction Dashboard

Proyek ini merupakan implementasi **Big Data Analytics (BDA)** untuk menganalisis data kecelakaan kendaraan bermotor di New York City menggunakan dataset **Motor Vehicle Collisions – Crashes**. Proyek ini berfokus pada pengolahan data berskala besar, analisis kecelakaan berdasarkan wilayah, prediksi tingkat risiko kecelakaan, serta penyajian hasil analisis melalui dashboard website.

Secara keseluruhan, proyek ini mengintegrasikan proses **preprocessing data, penyimpanan HDFS, Hadoop MapReduce, Machine Learning Random Forest, evaluasi model, dan dashboard website** untuk menghasilkan sistem analitik yang mampu membantu memahami pola kecelakaan serta memprediksi tingkat risiko kecelakaan kendaraan bermotor.

---

##  Fitur Utama

1. **Hadoop MapReduce (Borough Analysis)**: Pemrosesan paralel terdistribusi dengan Hadoop Streaming untuk mengagregasi total kecelakaan, korban luka (*injured*), dan korban meninggal (*killed*) berdasarkan wilayah (*Borough*).
2. **Risk Classification (Random Forest Classifier)**: Pemodelan Machine Learning untuk mengklasifikasikan tingkat keparahan risiko kecelakaan menjadi `Low`, `Medium`, atau `High` berdasarkan pola historis.
3. **Time-Series Forecasting**: Peramalan jumlah kecelakaan harian dan estimasi tingkat risikonya untuk 30 hari ke depan menggunakan regresi deret waktu dengan memperhitungkan faktor akhir pekan (*weekend multiplier*).
4. **Interactive Dashboard**: Visualisasi data interaktif yang disajikan dalam dua versi:
   - **Streamlit App** (`app.py`): Dashboard analitik berbasis Python Streamlit.
   - **Static Web Dashboard** (`website/`): Dashboard berbasis HTML, CSS, dan Javascript yang ringan dan cepat.

---

---



