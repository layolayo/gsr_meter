import hid
import time
import csv
import signal
import sys

# Constants from v16
VENDOR_ID = 0x1fc9
PRODUCT_ID = 0x0003
V_SOURCE = 6.371
R_REF = 83.0

data_log = []
running = True

def signal_handler(sig, frame):
    global running
    print("\nStopping...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

def main():
    print(f"Opening HID Device {hex(VENDOR_ID)}:{hex(PRODUCT_ID)}...")
    try:
        h = hid.device()
        h.open(VENDOR_ID, PRODUCT_ID)
        h.set_nonblocking(1) # Non-blocking for speed test
    except Exception as e:
        print(f"Failed to open HID: {e}")
        return

    print("Recording at ~60Hz... Press Ctrl+C to stop.")
    
    start_time = time.time()
    last_val = None
    changes = 0
    samples = 0
    
    while running:
        loop_start = time.time()
        
        # Read all available packets (drain buffer)
        # We want the LATEST packet
        packet = None
        while True:
            d = h.read(64)
            if d:
                packet = d
            else:
                break
        
        timestamp = time.time()
        
        ta = 0.0
        row = [timestamp, 0.0, 0.0] # Time, Delta, TA
        
        if packet and len(packet) >= 4 and packet[0] == 0x01:
            raw_val = (packet[2] << 8) | packet[3]
            voltage = raw_val / 10000.0
            
            ohms = 999999.9
            if voltage < (V_SOURCE - 0.005):
                try:
                    ohms = (voltage * R_REF) / (V_SOURCE - voltage)
                except: pass
            
            try:
                ta = (ohms * 1000 / (ohms * 1000 + 21250)) * 5.559 + 0.941
            except: 
                ta = 0.0
            
            # Check change
            if last_val is not None and ta != last_val:
                changes += 1
            last_val = ta
            
            row = [timestamp, timestamp - start_time, ta]
            data_log.append(row)
            samples += 1
        else:
            # No data? Log None or skip?
            # User wants 60Hz log. If no data, we log "No Data" or just skip.
            # Logging skip might hide connection drop.
            pass

        # Throttle to ~60Hz (16ms)
        elapsed = time.time() - loop_start
        sleep_time = 0.016 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        if samples % 60 == 0 and samples > 0:
            print(f"Recorded {samples} samples. Distinct Changes: {changes} ({(changes/samples)*100:.1f}%)")

    # Save
    fn = "GSR_Rate_Test.csv"
    print(f"Saving {len(data_log)} rows to {fn}...")
    with open(fn, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Elapsed", "TA"])
        writer.writerows(data_log)
    print("Done.")

if __name__ == "__main__":
    main()
