# Mengimpor library subprocess untuk menjalankan command Hadoop
import subprocess

# Menentukan lokasi file hasil preprocessing / data bersih
clean_file = r"D:\PROJEK BDA\output\motor_vehicle_clean.csv"

# Menentukan lokasi file risk prediction
risk_file = r"D:\PROJEK BDA\output\risk_prediction.csv"

# Menghapus file lama pada folder preprocessing di HDFS jika sudah ada
subprocess.run(
    'hdfs dfs -rm -r -f /bigdata/motor_vehicle/preprocessing/*',
    shell=True
)

# Menghapus file lama pada folder risk prediction di HDFS jika sudah ada
subprocess.run(
    'hdfs dfs -rm -r -f /bigdata/motor_vehicle/risk_prediction/*',
    shell=True
)

# Mengupload data hasil preprocessing ke HDFS
subprocess.run(
    f'hdfs dfs -put -f "{clean_file}" /bigdata/motor_vehicle/preprocessing/',
    shell=True
)

# Mengupload data risk prediction ke HDFS
subprocess.run(
    f'hdfs dfs -put -f "{risk_file}" /bigdata/motor_vehicle/risk_prediction/',
    shell=True
)

# Menampilkan isi folder preprocessing di HDFS
subprocess.run(
    'hdfs dfs -ls /bigdata/motor_vehicle/preprocessing',
    shell=True
)

# Menampilkan isi folder risk prediction di HDFS
subprocess.run(
    'hdfs dfs -ls /bigdata/motor_vehicle/risk_prediction',
    shell=True
)

# Menampilkan pesan bahwa proses upload selesai
print("Upload ke Hadoop HDFS selesai.")