import hid
import sys

# Target NXP Device
VID = 0x1fc9
PID = 0x0003

print(f"Searching for VID:0x{VID:04x} PID:0x{PID:04x}...")
print("-" * 60)

# 1. Enumerate and print EVERYTHING found
found_path = None
try:
    devices = hid.enumerate()
    for d in devices:
        print(f"FOUND: 0x{d['vendor_id']:04x}:0x{d['product_id']:04x} | {d['product_string']} | Path: {d['path']}")

        # Check if this is our target
        if d['vendor_id'] == VID and d['product_id'] == PID:
            found_path = d['path']

except Exception as e:
    print(f"Enumeration Error: {e}")
    sys.exit(1)

print("-" * 60)

# 2. Try to Open
if found_path:
    print(f"Target Found! Attempting to open by PATH: {found_path}")
    try:
        h = hid.device()
        h.open_path(found_path)
        print("\n[SUCCESS] Device Opened!")
        print(f"Manufacturer: {h.get_manufacturer_string()}")
        print(f"Product:      {h.get_product_string()}")
        h.close()
    except Exception as e:
        print(f"\n[FAILURE] Could not open path: {found_path}")
        print(f"Error Details: {e}")

        # Suggest Fix based on path
        if b"hidraw" in found_path:
            print("\nDiagnostic: Library is using hidraw (Correct).")
        else:
            print("\nDiagnostic: Library is NOT using hidraw. It might be using libusb.")
else:
    print("[FAILURE] Device not found in enumeration. Check physical connection.")