import csv
import sys

# Simulation of SessionData.load_session logic
class MockSessionData:
    def __init__(self):
        self.csv_data = []

    def load_session(self, csv_path):
        self.csv_data = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    try:
                        item = {
                            "t": float(row.get("Elapsed_Sec", 0)),
                            "ta": float(row.get("GSR_TA", 0)),
                        }
                        if item['ta'] > 0.1:
                            self.csv_data.append(item)
                    except: pass
            
            print(f"Loaded {len(self.csv_data)} rows.")
            
            if not self.csv_data: return
            
            # Normalize
            t_zero = self.csv_data[0]['t']
            print(f"t_zero raw: {t_zero}")
            
            if t_zero > 0.0:
                print(f"Normalizing by -{t_zero}")
                for row in self.csv_data:
                    row['t'] -= t_zero
                    
            # Check Monotonicity
            prev_t = -1.0
            disorder_count = 0
            gaps = []
            
            for i, row in enumerate(self.csv_data):
                t = row['t']
                if t < prev_t:
                    if disorder_count < 5:
                        print(f"DISORDER at row {i}: t={t:.4f} < prev={prev_t:.4f}")
                    disorder_count += 1
                
                if (t - prev_t) > 1.0 and prev_t != -1.0:
                     gaps.append(f"Gap at {prev_t:.2f}s -> {t:.2f}s (Delta: {t-prev_t:.2f}s)")
                
                prev_t = t
                
            if disorder_count > 0:
                print(f"TOTAL DISORDER EVENTS: {disorder_count}")
                print("CRITICAL: Bisect will fail on unsorted data.")
            else:
                print("Timestamps are Monotonic (Good).")
                
            if gaps:
                print(f"Found {len(gaps)} time gaps > 1.0s:")
                for g in gaps[:10]: print(g)
            else:
                print("No significant time gaps.")
                
        except Exception as e:
            print(f"Error: {e}")

mock = MockSessionData()
mock.load_session("Session_Data/GSR_Audio_2026-01-03_12-01-19.csv")
