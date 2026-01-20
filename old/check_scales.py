import pandas as pd
import os
import numpy as np

SESSION_DIR = "/home/matthew/PycharmProjects/BrainCo/Session_Data/Session_20260120_123019"
gsr_path = os.path.join(SESSION_DIR, "GSR.csv")
hrm_path = os.path.join(SESSION_DIR, "HRM.csv")

def analyze_col(name, series):
    if series is None or series.empty:
        print(f"--- {name}: NO DATA ---")
        return
    
    # Drop NaNs
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        print(f"--- {name}: ALL NAN ---")
        return

    # Stats
    mn, mx = s.min(), s.max()
    avg, std = s.mean(), s.std()
    p05 = s.quantile(0.05)
    p95 = s.quantile(0.95)
    
    print(f"--- {name} ---")
    print(f"  Range: [{mn:.4f}, {mx:.4f}]")
    print(f"  Mean:  {avg:.4f}  Std: {std:.4f}")
    print(f"  95% Range: [{p05:.4f}, {p95:.4f}]")
    
    # Suggest Scaler for +/- 6.0 Grid
    # Target is to map p95 to ~5.0 or 6.0
    abs_max = max(abs(p05), abs(p95))
    if abs_max > 0:
        s_6 = 6.0 / abs_max
        s_5 = 5.0 / abs_max
        print(f"  SUGGESTED SCALER (Target 6.0): {s_6:.2f}")
        print(f"  SUGGESTED SCALER (Target 5.0): {s_5:.2f}")

print(f"Analyzing Session: {SESSION_DIR}")

if os.path.exists(gsr_path):
    print("\nReading GSR...")
    try:
        df_gsr = pd.read_csv(gsr_path)
        if 'Delta_TA' in df_gsr.columns:
            analyze_col("GSR: Delta_TA", df_gsr['Delta_TA'])
        else:
            print("Delta_TA column missing in GSR.csv")
            # Calculate it simply?
            if 'TA' in df_gsr.columns:
                 # Emulate the app's Rolling Delta?
                 # App uses ~3-5s window.
                 print("(Approximating Delta from TA)")
                 # Just raw diff for now roughly
                 d = df_gsr['TA'].diff()
                 analyze_col("GSR: Calc_Delta_TA", d)
    except Exception as e:
        print(f"Error GSR: {e}")

if os.path.exists(hrm_path):
    print("\nReading HRM...")
    try:
        df_hrm = pd.read_csv(hrm_path)
        if 'Delta_HR' in df_hrm.columns:
            analyze_col("HRM: Delta_HR", df_hrm['Delta_HR'])
        else:
             print("Delta_HR missing")
             
        if 'Delta_HRV' in df_hrm.columns:
            analyze_col("HRM: Delta_HRV", df_hrm['Delta_HRV'])
        else:
             print("Delta_HRV missing")
             
    except Exception as e:
        print(f"Error HRM: {e}")
