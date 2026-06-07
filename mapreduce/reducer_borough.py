import sys

counts = {}

for line in sys.stdin:
    line = line.strip()

    if not line:
        continue

    parts = line.split("\t")

    if len(parts) != 2:
        continue

    borough = parts[0]
    value = parts[1]

    try:
        value = int(value)
    except:
        continue

    if borough not in counts:
        counts[borough] = 0

    counts[borough] += value

for borough in sorted(counts):
    print(borough + "\t" + str(counts[borough]))
