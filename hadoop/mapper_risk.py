#!/usr/bin/env python3
# mapper_risk.py
# Mapper untuk analisis tingkat risiko kecelakaan berdasarkan borough, bulan, hari, dan jam

import sys

for line in sys.stdin:
    line = line.strip()

    # Lewati header
    if line.startswith("borough"):
        continue

    fields = line.split(",")

    # Format risk_prediction.csv: borough, month, day_of_week, hour, crash_count, risk_level
    if len(fields) < 6:
        continue

    try:
        borough      = fields[0].strip()
        month        = fields[1].strip()
        day_of_week  = fields[2].strip()
        hour         = fields[3].strip()
        crash_count  = int(fields[4].strip())
        risk_level   = fields[5].strip()

        key = f"{borough},{month},{day_of_week},{hour}"

        # Output: key \t crash_count,risk_level
        print(f"{key}\t{crash_count},{risk_level}")

    except (ValueError, IndexError):
        continue
