import csv

fn = "Session_Data/GSR_Audio_2026-01-03_12-01-19.csv"

vals = []
reader_list = []
try:
    with open(fn, 'r') as f:
        reader = csv.DictReader(f)
        reader_list = list(reader) # Convert to list for multiple passes
        for row in reader_list:
            try:
                v = float(row.get("GSR_TA", 0))
                vals.append(v)
            except: pass
except FileNotFoundError:
    print("File not found.")
    exit()

if not vals:
    print("No data found.")
    exit()

# Analyze Jumps
jumps = []
prev = vals[0]
max_jump = 0
max_jump_idx = -1

for i, v in enumerate(vals[1:]):
    diff = abs(v - prev)
    if diff > max_jump:
        max_jump = diff
        max_jump_idx = i + 1
    prev = v

print(f"File: {fn}")
print(f"Total Rows: {len(vals)}")
print(f"Max TA: {max(vals):.4f}")
print(f"Min TA: {min(vals):.4f}")
print(f"Max Jump between samples: {max_jump:.6f} at index {max_jump_idx}")
print(f" - Value before: {vals[max_jump_idx-1]:.4f}")
print(f" - Value after:  {vals[max_jump_idx]:.4f}")

if max_jump > 0.5:
    print("RESULT: FAIL - Sudden Jump Detected (>0.5)")
elif max_jump > 0.1:
    print("RESULT: WARNING - Moderate Jump Detected (>0.1)")
else:
    print("RESULT: PASS - Data is Smooth (<0.1 per sample)")

# Analyze SetPoint (Center)
centers = [float(row.get("GSR_SetPoint", 0)) for row in reader_list if row.get("GSR_SetPoint")]
if centers:
    min_c = min(centers)
    max_c = max(centers)
    print(f"\nSetPoint (TA SET) Analysis:")
    print(f"Min: {min_c:.4f}")
    print(f"Max: {max_c:.4f}")
    if (max_c - min_c) > 0.001:
        print("WARNING: SetPoint VARIES in file!")
    else:
        print("SetPoint is CONSTANT.")
