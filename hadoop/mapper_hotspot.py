#!/usr/bin/env python3
# mapper_hotspot.py
# Mapper untuk analisis hotspot kecelakaan berdasarkan koordinat

import sys

for line in sys.stdin:
    line = line.strip()

    # Lewati header
    if line.startswith("crash_date"):
        continue

    fields = line.split(",")

    # Pastikan jumlah kolom cukup
    if len(fields) < 7:
        continue

    try:
        latitude  = float(fields[3].strip())
        longitude = float(fields[4].strip())

        # Bulatkan koordinat ke 2 desimal untuk pengelompokan area
        lat_rounded = round(latitude, 2)
        lon_rounded = round(longitude, 2)

        key = f"{lat_rounded},{lon_rounded}"

        # Output: koordinat \t 1
        print(f"{key}\t1")

    except (ValueError, IndexError):
        continue
