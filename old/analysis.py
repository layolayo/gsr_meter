import csv
import numpy as np
import os

# Paths
SESSION_DIR = "Session_Data/Session_2026-01-04_13-05-15"
EEG_FILE = os.path.join(SESSION_DIR, "eeg.csv")
GSR_FILE = os.path.join(SESSION_DIR, "gsr.csv")

def read_csv_columns(fn):
    data = {}
    with open(fn, 'r') as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        for h in headers: data[h] = []
        for row in reader:
            if not row: continue
            for i, h in enumerate(headers):
                try: val = float(row[i])
                except: val = 0.0
                data[h].append(val)
    # Convert to numpy arrays
    for k in data: data[k] = np.array(data[k])
    return data

def analyze_correlations():
    if not os.path.exists(EEG_FILE):
        print(f"File not found: {EEG_FILE}")
        return

    print("Loading Data (No Pandas)...")
    eeg = read_csv_columns(EEG_FILE)
    gsr = read_csv_columns(GSR_FILE)
    
    # Check if empty
    if len(eeg['Timestamp']) == 0 or len(gsr['Timestamp']) == 0:
        print("Data empty.")
        return

    # Time Alignment
    t_start = max(eeg['Timestamp'][0], gsr['Timestamp'][0])
    t_end = min(eeg['Timestamp'][-1], gsr['Timestamp'][-1])
    
    # Create 10Hz common timebase
    t_common = np.arange(t_start, t_end, 0.1)
    
    print(f"Analyzing {len(t_common)} samples (10Hz) over {t_end - t_start:.1f}s")
    
    # Interpolate
    focus = np.interp(t_common, eeg['Timestamp'], eeg['EEG_Focus'])
    calm = np.interp(t_common, eeg['Timestamp'], eeg['EEG_Calm'])
    
    # Bands
    alpha = np.interp(t_common, eeg['Timestamp'], eeg['EEG_Alpha'])
    beta = np.interp(t_common, eeg['Timestamp'], eeg['EEG_Beta'])
    delta = np.interp(t_common, eeg['Timestamp'], eeg['EEG_Delta'])
    theta = np.zeros_like(t_common)
    gamma = np.interp(t_common, eeg['Timestamp'], eeg['EEG_Gamma'])
    
    # Check headers for Theta (sometimes variable)
    if 'EEG_Theta' in eeg:
         theta = np.interp(t_common, eeg['Timestamp'], eeg['EEG_Theta'])
    
    gsr_ta = np.interp(t_common, gsr['Timestamp'], gsr['GSR_TA'])
    
    # Correlation
    # Stack matrix: [Focus, Calm, Delta, Theta, Alpha, Beta, Gamma, TA]
    stack = np.vstack([focus, calm, delta, theta, alpha, beta, gamma, gsr_ta])
    corr_mat = np.corrcoef(stack)
    
    labels = ["Focus", "Calm", "Delta", "Theta", "Alpha", "Beta", "Gamma", "GSR_TA"]
    
    print("\n--- Correlation with GSR_TA (Pearson) ---")
    print(f"{'Metric':<10} | {'Correlation':<12} | {'Interpretation'}")
    print("-" * 45)
    
    for i in range(7):
        c = corr_mat[i, 7]
        interp = ""
        if abs(c) > 0.5: interp = "STRONG"
        elif abs(c) > 0.3: interp = "Moderate"
        else: interp = "Weak"
        
        print(f"{labels[i]:<10} | {c: .4f}      | {interp}")
        
    
    
    # --- CONFLICT vs RELIEF PATTERN DETECTION ---
    print("\n\n" + "="*70)
    print("PATTERN DETECTION: CONFLICT vs RELIEF")
    print("="*70)
    print("\nCONFLICT: Rising TA → Flicker/Drop → TA stays HIGH (no release)")
    print("RELIEF:  High TA → Sudden Drop → TA settles to NEW LOWER level")
    
    # === PARAMETERS ===
    TREND_WINDOW = 50          # 5 seconds at 10Hz
    RISING_MIN_SLOPE = 0.0003  # Minimum slope
    RISING_MIN_DURATION = 30   # 3 seconds minimum rising phase
    DROP_SIGMA_MULT = 1.5      # Flicker threshold
    RELIEF_DROP_MULT = 3.0     # Relief needs STRONG drop (3σ)
    LEVEL_CHECK_WINDOW = 20    # 2 seconds for level comparison
    EVENT_WINDOW = 20          # ±2 seconds for brainwave extraction
    MIN_EVENT_GAP = 100        # 10 seconds between events
    RELIEF_RECOVERY_PCT = 1.7  # Relief must recover 170%+ of the rise
    
    # 1. Background Trend
    kernel = np.ones(TREND_WINDOW) / TREND_WINDOW
    ta_trend = np.convolve(gsr_ta, kernel, mode='same')
    trend_slope = np.gradient(ta_trend)
    
    # 2. Instantaneous Change & Noise Floor
    d_ta = np.diff(gsr_ta, prepend=gsr_ta[0])
    noise_floor = np.std(d_ta)
    flicker_threshold = -noise_floor * DROP_SIGMA_MULT
    relief_threshold = -noise_floor * RELIEF_DROP_MULT
    
    print(f"\nNoise floor (σ): {noise_floor:.6f}")
    print(f"Flicker threshold ({DROP_SIGMA_MULT}σ): {flicker_threshold:.6f}")
    print(f"Relief threshold ({RELIEF_DROP_MULT}σ): {relief_threshold:.6f}")
    
    # 3. Identify Rising Phases
    is_rising = trend_slope > RISING_MIN_SLOPE
    rising_phases = []
    
    i = TREND_WINDOW
    while i < len(is_rising) - LEVEL_CHECK_WINDOW:
        if is_rising[i]:
            start = i
            while i < len(is_rising) and is_rising[i]:
                i += 1
            end = i
            if (end - start) >= RISING_MIN_DURATION:
                rising_phases.append((start, end))
        else:
            i += 1
    
    print(f"Rising phases detected: {len(rising_phases)}")
    
    # 4. Detect BOTH patterns
    conflict_events = []  # Flicker but stayed high
    relief_events = []    # Genuine drop to new level
    
    for phase_start, phase_end in rising_phases:
        # Get baseline and peak
        pre_level = np.mean(gsr_ta[max(0, phase_start - LEVEL_CHECK_WINDOW):phase_start])
        peak_level = np.max(gsr_ta[phase_start:phase_end])
        rise_amount = peak_level - pre_level
        
        if rise_amount <= 0:
            continue
        
        # Search for drops during/after rise
        search_start = phase_start + RISING_MIN_DURATION // 2
        search_end = min(phase_end + 50, len(d_ta) - LEVEL_CHECK_WINDOW)  # 5s after phase
        
        for idx in range(search_start, search_end):
            if d_ta[idx] < flicker_threshold:
                post_level = np.mean(gsr_ta[idx:idx + LEVEL_CHECK_WINDOW])
                drop_from_peak = peak_level - post_level
                recovery_pct = drop_from_peak / rise_amount if rise_amount > 0 else 0
                
                # Check gap from previous events
                conflict_gap_ok = not conflict_events or (idx - conflict_events[-1][0]) > MIN_EVENT_GAP
                relief_gap_ok = not relief_events or (idx - relief_events[-1][0]) > MIN_EVENT_GAP
                
                # RELIEF: Strong drop AND recovers 50%+ of rise (settles significantly lower)
                if d_ta[idx] < relief_threshold and recovery_pct >= RELIEF_RECOVERY_PCT:
                    if relief_gap_ok:
                        relief_events.append((idx, d_ta[idx], peak_level, post_level, recovery_pct))
                
                # CONFLICT: Flicker during rising trend but stays near peak (recovery < 30%)
                elif recovery_pct < 0.3 and trend_slope[idx] > 0:
                    if conflict_gap_ok:
                        conflict_events.append((idx, d_ta[idx], peak_level, post_level, recovery_pct))
    
    print(f"\nCONFLICT events (flicker, stayed high): {len(conflict_events)}")
    print(f"RELIEF events (genuine release): {len(relief_events)}")
    
    # 5. Extract & Compare Brainwave Signatures
    b_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    all_bands = [delta, theta, alpha, beta, gamma]
    baseline = [np.mean(b) for b in all_bands]
    
    def analyze_events(events, label):
        if len(events) == 0:
            print(f"\nNo {label} events detected.")
            return None
        
        print(f"\n" + "-"*70)
        print(f"{label.upper()} EVENTS ({len(events)} found)")
        print("-"*70)
        print(f"{'#':<3} | {'Time':<7} | {'Drop':<9} | {'Peak→Post':<14} | {'Recovery':<8} | {'Delta':<6} | {'Theta':<6} | {'Alpha':<6} | {'Beta':<6} | {'Gamma':<6}")
        print("-"*100)
        
        event_data = []
        for i, (drop_idx, drop_mag, peak, post, recov) in enumerate(events):
            start = max(0, drop_idx - EVENT_WINDOW)
            end = min(len(delta), drop_idx + EVENT_WINDOW)
            band_avgs = [np.mean(b[start:end]) for b in all_bands]
            event_data.append(band_avgs)
            
            elapsed = t_common[drop_idx] - t_common[0]
            print(f"{i+1:<3} | {elapsed:<7.1f} | {drop_mag:<9.5f} | {peak:.3f}→{post:.3f} | {recov*100:<7.0f}% | {band_avgs[0]:<6.1f} | {band_avgs[1]:<6.1f} | {band_avgs[2]:<6.1f} | {band_avgs[3]:<6.1f} | {band_avgs[4]:<6.1f}")
        
        return np.array(event_data)
    
    conflict_data = analyze_events(conflict_events, "CONFLICT")
    relief_data = analyze_events(relief_events, "RELIEF")
    
    # 6. Comparative Summary
    print("\n" + "="*70)
    print("BRAINWAVE SIGNATURE COMPARISON")
    print("="*70)
    print(f"\n{'Band':<10} | {'Baseline':<10} | {'CONFLICT':<12} | {'RELIEF':<12} | {'Difference':<12} | Interpretation")
    print("-"*90)
    
    for i, name in enumerate(b_names):
        base_val = baseline[i]
        conflict_mean = np.mean(conflict_data[:, i]) if conflict_data is not None and len(conflict_data) > 0 else 0
        relief_mean = np.mean(relief_data[:, i]) if relief_data is not None and len(relief_data) > 0 else 0
        
        diff = relief_mean - conflict_mean if conflict_mean > 0 else 0
        diff_pct = (diff / conflict_mean * 100) if conflict_mean > 0 else 0
        
        interp = ""
        if diff_pct > 10:
            interp = "↑ Higher in Relief"
        elif diff_pct < -10:
            interp = "↓ Higher in Conflict"
        else:
            interp = "~Similar"
        
        print(f"{name:<10} | {base_val:<10.1f} | {conflict_mean:<12.1f} | {relief_mean:<12.1f} | {diff_pct:>+10.1f}% | {interp}")
    
    # 7. Detection Criteria Summary
    print("\n" + "="*70)
    print("REAL-TIME DETECTION CRITERIA")
    print("="*70)
    
    print("\n🔴 CONFLICT (Internal Resistance):")
    print(f"   GSR: Rising trend + flicker (>{abs(flicker_threshold):.5f}) BUT post-level stays near peak (<30% recovery)")
    if conflict_data is not None and len(conflict_data) > 0:
        print(f"   Brainwaves at Conflict:")
        for i, name in enumerate(b_names):
            m = np.mean(conflict_data[:, i])
            s = np.std(conflict_data[:, i])
            print(f"     {name}: {m:.1f} ± {s:.1f}")
    
    print("\n🟢 RELIEF (New Understanding):")
    print(f"   GSR: High TA + strong drop (>{abs(relief_threshold):.5f}) + settles ≥{RELIEF_RECOVERY_PCT*100:.0f}% below peak")
    if relief_data is not None and len(relief_data) > 0:
        print(f"   Brainwaves at Relief:")
        for i, name in enumerate(b_names):
            m = np.mean(relief_data[:, i])
            s = np.std(relief_data[:, i])
            print(f"     {name}: {m:.1f} ± {s:.1f}")

if __name__ == "__main__":
    analyze_correlations()
