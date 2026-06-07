#!/usr/bin/env python3
# reducer_risk.py
# Reducer untuk analisis tingkat risiko kecelakaan berdasarkan borough, bulan, hari, dan jam

import sys

current_key   = None
total_count   = 0
risk_label    = "Low Risk"

for line in sys.stdin:
    line = line.strip()
    parts = line.split("\t")

    if len(parts) != 2:
        continue

    key, value = parts[0], parts[1]

    value_parts = value.split(",")
    if len(value_parts) < 2:
        continue

    try:
        crash_count = int(value_parts[0])
        risk_level  = value_parts[1].strip()
    except ValueError:
        continue

    if current_key == key:
        total_count += crash_count
        # Jika ada label High Risk, dominankan
        if risk_level == "High Risk":
            risk_label = "High Risk"
    else:
        if current_key is not None:
            print(f"{current_key}\t{total_count}\t{risk_label}")
        current_key = key
        total_count = crash_count
        risk_label  = risk_level

# Output baris terakhir
if current_key is not None:
    print(f"{current_key}\t{total_count}\t{risk_label}")
