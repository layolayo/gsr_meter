import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os
import sys

# --- Configuration ---
SESSION_ROOT = "/home/matthew/PycharmProjects/BrainCo/Session_Data"

def get_latest_session():
    """Finds the most recently created Session directory."""
    all_sessions = glob.glob(os.path.join(SESSION_ROOT, "Session_*"))
    if not all_sessions:
        return None
    latest = max(all_sessions, key=os.path.getctime)
    return latest

def load_data(session_path):
    """Loads GSR and HRM data from the session folder."""
    gsr_path = os.path.join(session_path, "GSR.csv")
    hrm_path = os.path.join(session_path, "HRM.csv")
    
    df_gsr = None
    df_hrm = None
    
    if os.path.exists(gsr_path):
        try:
            df_gsr = pd.read_csv(gsr_path)
            # Parse Timestamp (HH:MM:SS.ffffff)
            # We assume they are on the same day for relative plotting
            df_gsr['dt'] = pd.to_datetime(df_gsr['Timestamp'], format='%H:%M:%S.%f')
            # Create relative seconds
            start_time = df_gsr['dt'].iloc[0]
            df_gsr['Seconds'] = (df_gsr['dt'] - start_time).dt.total_seconds()
        except Exception as e:
            print(f"Error loading GSR: {e}")

    if os.path.exists(hrm_path):
        try:
            df_hrm = pd.read_csv(hrm_path)
            if not df_hrm.empty:
                 # Check if headers are duplicated (common issue in streaming CSVs)
                 df_hrm = df_hrm[df_hrm['Timestamp'] != 'Timestamp']
                 df_hrm['dt'] = pd.to_datetime(df_hrm['Timestamp'], format='%H:%M:%S.%f')
                 # Align to GSR start if possible, else self-relative
                 start_time = df_gsr['dt'].iloc[0] if df_gsr is not None else df_hrm['dt'].iloc[0]
                 df_hrm['Seconds'] = (df_hrm['dt'] - start_time).dt.total_seconds()
                 
                 # Convert numeric cols
                 df_hrm['HR_BPM'] = pd.to_numeric(df_hrm['HR_BPM'])
                 df_hrm['Z_HRV'] = pd.to_numeric(df_hrm['Z_HRV'], errors='coerce')
                 if 'RMSSD_MS' in df_hrm.columns:
                    df_hrm['RMSSD_MS'] = pd.to_numeric(df_hrm['RMSSD_MS'], errors='coerce')
        except Exception as e:
            print(f"Error loading HRM: {e}")
            
    return df_gsr, df_hrm

def plot_session(df_gsr, df_hrm, save_path):
    """Generates the analysis plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Plot 1: GSR & Patterns ---
    if df_gsr is not None and not df_gsr.empty:
        ax1.plot(df_gsr['Seconds'], df_gsr['TA'], label='TA (Conductance)', color='blue', linewidth=1.5)
        ax1.set_ylabel("GSR TA Value")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("GSR Trend & Patterns")
        
        # Highlight Major Events (Notes)
        if 'Notes' in df_gsr.columns:
            events = df_gsr[df_gsr['Notes'].notna() & (df_gsr['Notes'] != '')]
            
            # Filter out minor messages if needed? 
            # For now, plot all notes as requested "major incidents"
            
            for idx, row in events.iterrows():
                note = row['Notes']
                # Skip Motion Lock logs if frequent, but grep showed they are rare/important
                
                # Vertical Line
                ax1.axvline(x=row['Seconds'], color='black', linestyle='--', alpha=0.5)
                
                # Label at top
                # Use transAxes for y-position (top of chart)
                ax1.text(row['Seconds'], 1.05, note, 
                         transform=ax1.get_xaxis_transform(),
                         rotation=45, ha='left', fontsize=9, color='black', weight='bold')

        # Highlight Motion (Keep as scatter on line)
        if 'Motion' in df_gsr.columns:
            motion_pts = df_gsr[df_gsr['Motion'] == 1]
            if not motion_pts.empty:
                ax1.scatter(motion_pts['Seconds'], motion_pts['TA'], color='orange', s=10, label='Motion Lock')

        ax1.legend(loc='upper left')

    # --- Plot 2: HRM & States ---
    if df_hrm is not None and not df_hrm.empty:
        # Primary Axis: Heart Rate
        l1 = ax2.plot(df_hrm['Seconds'], df_hrm['HR_BPM'], label='Heart Rate (BPM)', color='green', linewidth=1.5)
        ax2.set_ylabel("BPM", color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_xlabel("Time (Seconds)")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Heart Rate & HRV (Biofeedback State)")
        
        # Secondary Axis: HRV
        if 'RMSSD_MS' in df_hrm.columns:
             ax2t = ax2.twinx()
             # Filter huge outliers or 0
             hrv_data = df_hrm['RMSSD_MS']
             l2 = ax2t.plot(df_hrm['Seconds'], hrv_data, label='HRV (RMSSD)', color='purple', alpha=0.6, linewidth=1.0)
             ax2t.set_ylabel("HRV (ms)", color='purple')
             ax2t.tick_params(axis='y', labelcolor='purple')
             
             # Combined Legend
             lns = l1 + l2
             labs = [l.get_label() for l in lns]
             ax2.legend(lns, labs, loc='upper left')
        else:
             ax2.legend(loc='upper left')
        
        # Color Background by State/Quadrant
        if 'Quadrant' in df_hrm.columns:
            # Iterate through state changes to draw spans
            hrm_sorted = df_hrm.sort_values('Seconds')
            hrm_sorted['Quadrant'] = hrm_sorted['Quadrant'].astype(str)
            
            changes = hrm_sorted[hrm_sorted['Quadrant'] != hrm_sorted['Quadrant'].shift()]
            
            q_colors = {
                'Rest': 'whitesmoke',
                'Focus': 'lightcyan',
                'Calm': 'lavender',
                'Engage': 'mistyrose',
                'nan': 'white'
            }
            
            prev_time = hrm_sorted['Seconds'].iloc[0]
            prev_q = hrm_sorted['Quadrant'].iloc[0]
            
            for idx, row in changes.iloc[1:].iterrows():
                t = row['Seconds']
                q = row['Quadrant']
                c = q_colors.get(prev_q, 'white')
                if 'Rest' in prev_q: c = 'whitesmoke' # Fuzzy match
                
                ax2.axvspan(prev_time, t, color=c, alpha=0.5)
                prev_time = t
                prev_q = q
            
            # Last span
            ax2.axvspan(prev_time, hrm_sorted['Seconds'].iloc[-1], color=q_colors.get(prev_q, 'white'), alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Analysis saved to: {save_path}")

if __name__ == "__main__":
    latest_dir = get_latest_session()
    if not latest_dir:
        print("No sessions found.")
        sys.exit(1)
        
    print(f"Analyzing Session: {latest_dir}")
    df_g, df_h = load_data(latest_dir)
    
    out_file = os.path.join(latest_dir, "analysis_result.png")
    plot_session(df_g, df_h, out_file)
    
    # Also save one to project root for easy access
    plot_session(df_g, df_h, "latest_analysis.png")
