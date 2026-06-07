# Mengimpor library untuk membaca input dari Hadoop Streaming
import sys
import csv

# Membaca file CSV dari standard input
reader = csv.DictReader(sys.stdin)

# Membaca setiap baris data
for row in reader:

    # Mengambil nilai kolom borough sesuai header dataset
    borough = row.get("borough", "").strip()

    # Jika borough tidak kosong, kirim ke reducer dengan nilai 1
    if borough:
        print(f"{borough}\t1")
