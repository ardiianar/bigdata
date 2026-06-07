#  Motor Vehicle Collision Big Data Analytics & Forecasting Dashboard

Sebuah proyek **Big Data Analytics (BDA)** dan **Machine Learning** untuk menganalisis, mengklasifikasikan tingkat risiko, dan meramalkan tingkat kecelakaan kendaraan bermotor di New York City (NYC) menggunakan dataset skala besar dari *NYC Open Data*. Proyek ini mengintegrasikan ekosistem **Hadoop MapReduce** untuk pemrosesan paralel terdistribusi dan model **Random Forest Classifier** untuk klasifikasi & peramalan prediktif.

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



