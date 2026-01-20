
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os

# --- CONSTANTS ---
FS = 250
FFT_WINDOW_SEC = 1.0  # 1Hz resolution (Physics)
TREND_WINDOW_SEC = 5.0 # Display Smoothing
SESSION_DIR = "Session_Data/Session_2026-01-04_13-05-15"
EEG_FILE = os.path.join(SESSION_DIR, "eeg.csv")
GSR_FILE = os.path.join(SESSION_DIR, "gsr.csv")

# Filter Setup (From v26)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b_filt, a_filt = butter_bandpass(0.5, 50.0, FS, order=5)

def calculate_relative_bands(data_segment):
    # FFT
    data_np = np.array(data_segment)
    fft_vals = np.abs(np.fft.rfft(data_np))
    fft_freq = np.fft.rfftfreq(len(data_np), 1.0/FS)
    
    # Define Bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta':  (13, 30),
        'Gamma': (30, 50)
    }
    
    powers = {}
    total_power = 0
    
    for band, (low, high) in bands.items():
        idx = np.where((fft_freq >= low) & (fft_freq <= high))[0]
        if len(idx) > 0:
            band_power = np.sum(fft_vals[idx])
            powers[band] = band_power
            total_power += band_power
            
    # Normalize to percentages
    rel_powers = []
    if total_power > 0:
        for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
            rel_powers.append((powers.get(band, 0) / total_power) * 100)
    else:
        rel_powers = [0, 0, 0, 0, 0]
        
    return rel_powers

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    print(f"Loading Session: {SESSION_DIR}")
    
    # 1. LOAD EEG RAW
    raw_timestamps = []
    raw_eeg_vals = []
    
    with open(EEG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                raw_timestamps.append(float(row['Timestamp']))
                raw_eeg_vals.append(float(row['EEG_Raw']))
            except: pass
            
    raw_eeg = np.array(raw_eeg_vals)
    ts_eeg = np.array(raw_timestamps)
    
    print(f"Loaded {len(raw_eeg)} EEG samples.")
    
    # 2. FILTER EEG
    print("Filtering EEG (0.5-50Hz)...")
    filtered_eeg = lfilter(b_filt, a_filt, raw_eeg)
    
    # 3. COMPUTE BANDS (1.0s Window)
    print("Computing Bands (FFT Window = 1.0s)...")
    window_samples = int(FS * FFT_WINDOW_SEC)
    step_samples = int(FS * 0.1) # 10Hz update rate (simulate live)
    
    band_ts = []
    band_data = [] # [Delta, Theta, Alpha, Beta, Gamma]
    
    for i in range(window_samples, len(filtered_eeg), step_samples):
        segment = filtered_eeg[i-window_samples:i]
        bands = calculate_relative_bands(segment)
        
        # Timestamp is end of window
        band_ts.append(ts_eeg[i])
        band_data.append(bands)
        
    band_ts = np.array(band_ts)
    band_data = np.array(band_data) # Shape (N, 5)
    
    print(f"Computed {len(band_data)} band snapshots.")
    
    # 4. APPLY 5s TREND SMOOTHING
    print(f"Applying Trend Smoothing (Window = {TREND_WINDOW_SEC}s)...")
    # 5 seconds at 10Hz update rate = 50 samples
    smooth_window = int(TREND_WINDOW_SEC / 0.1) 
    
    smoothed_bands = np.zeros_like(band_data)
    for b in range(5):
        smoothed = moving_average(band_data[:, b], n=smooth_window)
        # Pad beginning to match length
        pad_len = len(band_data) - len(smoothed)
        # Pad with first value (or 0)
        smoothed_bands[:, b] = np.pad(smoothed, (pad_len, 0), mode='edge')
        
    # 5. LOAD GSR
    print("Loading GSR...")
    gsr_ts = []
    gsr_vals = []
    with open(GSR_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gsr_ts.append(float(row['Timestamp']))
                gsr_vals.append(float(row['GSR_TA']))
            except: pass
            
    gsr_ts = np.array(gsr_ts)
    gsr_vals = np.array(gsr_vals)
    
    # 6. CORRELATION ANALYSIS
    print("Calculating Correlations...")
    # Common Timebase (10Hz)
    t_start = max(band_ts[0], gsr_ts[0])
    t_end = min(band_ts[-1], gsr_ts[-1])
    t_common = np.arange(t_start, t_end, 0.1)
    
    # Interpolate to common timebase
    bands_interp = []
    for i in range(5):
        bands_interp.append(np.interp(t_common, band_ts, smoothed_bands[:, i]))
    bands_interp = np.array(bands_interp)
    
    gsr_interp = np.interp(t_common, gsr_ts, gsr_vals)
    
    # Calculate Pearson Correlation
    print("\n--- Correlation w/ GSR (Pearson) ---")
    print(f"{'Band':<10} | {'Corr':<8} | {'Interp'}")
    print("-" * 35)
    
    labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    for i in range(5):
        corr = np.corrcoef(bands_interp[i], gsr_interp)[0, 1]
        interp = "Weak"
        if abs(corr) > 0.5: interp = "STRONG"
        elif abs(corr) > 0.3: interp = "Moderate"
        
        print(f"{labels[i]:<10} | {corr:.4f}   | {interp}")
    print("-" * 35 + "\n")

    # 7. PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot Bands
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    for i in range(5):
        ax1.plot(band_ts, smoothed_bands[:, i], label=labels[i], color=colors[i], linewidth=1.5)
        
    ax1.set_ylabel('Relative Power (%)')
    ax1.set_title(f'Smoothed EEG Trends (FFT=1s, Trend={TREND_WINDOW_SEC}s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 60) # Typical range
    
    # Plot GSR
    ax2.plot(gsr_ts, gsr_vals, color='orange', linewidth=2, label='GSR TA')
    ax2.set_ylabel('GSR TA (uS)')
    ax2.set_xlabel('Time (Unix Timestamp)')
    ax2.set_title('GSR Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v27_trend_analysis.png')
    print("Saved plot to v27_trend_analysis.png")
    plt.show()

if __name__ == "__main__":
    main()
