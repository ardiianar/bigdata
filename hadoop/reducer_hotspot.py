#!/usr/bin/env python3
# reducer_hotspot.py
# Reducer untuk analisis hotspot kecelakaan berdasarkan koordinat

import sys

current_coord = None
current_count = 0

for line in sys.stdin:
    line = line.strip()
    parts = line.split("\t")

    if len(parts) != 2:
        continue

    coord, count = parts[0], parts[1]

    try:
        count = int(count)
    except ValueError:
        continue

    if current_coord == coord:
        current_count += count
    else:
        if current_coord is not None:
            print(f"{current_coord}\t{current_count}")
        current_coord = coord
        current_count = count

# Output baris terakhir
if current_coord is not None:
    print(f"{current_coord}\t{current_count}")
