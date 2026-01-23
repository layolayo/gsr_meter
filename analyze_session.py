import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
import base64
import io
import webbrowser

# --- Configuration ---
SESSION_ROOT = "/home/matthew/PycharmProjects/gsr_meter/Session_Data"

def list_sessions():
    """Lists all available session directories."""
    all_dirs = [d for d in glob.glob(os.path.join(SESSION_ROOT, "*")) if os.path.isdir(d)]
    valid_sessions = [d for d in all_dirs if os.path.exists(os.path.join(d, "GSR.csv"))]
    return sorted(valid_sessions, key=os.path.getctime, reverse=True)

def select_session():
    """CLI to select which session to analyze."""
    sessions = list_sessions()
    if not sessions:
        print("No sessions found.")
        return None
    
    print("\nAvailable GSR Sessions:")
    for idx, s in enumerate(sessions):
        print(f"[{idx}] {os.path.basename(s)}")
    
    try:
        choice = input(f"\nSelect session [0-{len(sessions)-1}] (default 0): ").strip()
        if not choice: return sessions[0]
        return sessions[int(choice)]
    except (ValueError, IndexError):
        print("Invalid choice, defaulting to latest.")
        return sessions[0]

def load_data(session_path):
    """Loads GSR data and identifies calibration points."""
    gsr_path = os.path.join(session_path, "GSR.csv")
    
    if not os.path.exists(gsr_path):
        print(f"GSR file not found: {gsr_path}")
        return None

    try:
        # Load GSR
        df = pd.read_csv(gsr_path, low_memory=False)
        
        # Parse Timestamp and Elapsed relative to start
        if 'Elapsed' in df.columns:
            if df['Elapsed'].dtype == object and ':' in str(df['Elapsed'].iloc[0]):
                def parse_time(time_str):
                    try:
                        pts = str(time_str).split(':')
                        return float(pts[0])*3600 + float(pts[1])*60 + float(pts[2])
                    except: return np.nan
                df['Seconds'] = df['Elapsed'].apply(parse_time)
            else:
                df['Seconds'] = pd.to_numeric(df['Elapsed'], errors='coerce')
        else:
            df['dt'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f', errors='coerce')
            df['Seconds'] = (df['dt'] - df['dt'].iloc[0]).dt.total_seconds()

        # Ensure numeric types
        for col in ['TA', 'TA Counter', 'Window_Size']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        print(f"Error loading GSR: {e}")
        return None

def analyze_incidents(df):

    if 'Pattern' not in df.columns:
        return [], [], []

    # 1. Smooth TA for extrema detection
    df['TA_Smooth'] = df['TA'].rolling(window=3, center=True).mean().fillna(df['TA'])
     
    incidents = []
    current_group_indices = []
    current_direction = None # "DOWN" or "UP"
    
    # Define directional patterns
    down_patterns = ["FALL", "SHORT FALL", "LONG FALL", "BLOWDOWN", "TICK"]
    up_patterns = ["RISE", "SHORT RISE", "LONG RISE", "ROCKET READ"]
    
    # Identify groups of consecutive movement patterns
    for i, row in df.iterrows():
        pat = str(row['Pattern']).strip().upper() if pd.notna(row['Pattern']) else ""
        
        direction = None
        if pat in down_patterns: direction = "DOWN"
        elif pat in up_patterns: direction = "UP"
        
        if direction and (current_direction is None or direction == current_direction):
            current_group_indices.append(i)
            current_direction = direction
        else:
            if current_group_indices:
                inc = refined_incident_record(df, current_group_indices, current_direction)
                if inc: incidents.append(inc)
                current_group_indices = []
            
            if direction:
                current_group_indices.append(i)
                current_direction = direction
            else:
                current_direction = None
                
    if current_group_indices:
        inc = refined_incident_record(df, current_group_indices, current_direction)
        if inc: incidents.append(inc)
        
    # FINAL FILTER: Only Significant Patterns (as requested by user)
    significant = ["BLOWDOWN", "LONG FALL", "LONG RISE", "ROCKET READ"]
    incidents = [i for i in incidents if i['pattern'] in significant]
        
    # 4. Floating Wave Detector (Rhythmic Oscillations)
    # High-pass filter (simple detrend)
    df['TA_Detrend'] = df['TA'] - df['TA'].rolling(window=30).mean()
    # Check for rhythmic peaks (0.2Hz - 1.0Hz)
    # At 10Hz, this is one peak every 10-50 samples
    df['Is_Wave'] = False
    for start in range(0, len(df)-50, 10):
        window = df['TA_Detrend'].iloc[start:start+50]
        # Identify local extrema
        peaks = (window > window.shift(1)) & (window > window.shift(-1))
        peak_count = peaks.sum()
        if 2 <= peak_count <= 5: # Typical for 5 seconds at these frequencies
            # Check for regularity (approximate)
            peak_indices = np.where(peaks)[0]
            if len(peak_indices) >= 3:
                intervals = np.diff(peak_indices)
                if np.std(intervals) / np.mean(intervals) < 0.4: # Relaxed CV (0.3 -> 0.4) to catch more End Phenomena
                    df.loc[start:start+50, 'Is_Wave'] = True

    wave_zones = []
    current_wave = None
    for i, row in df.iterrows():
        if row['Is_Wave']:
            if current_wave is None:
                current_wave = {'start': row['Seconds'], 'end': row['Seconds']}
            else:
                current_wave['end'] = row['Seconds']
        else:
            if current_wave:
                if current_wave['end'] - current_wave['start'] > 4.0: # At least 2 full cycles
                    wave_zones.append(current_wave)
                current_wave = None
    if current_wave: wave_zones.append(current_wave)

    return incidents, wave_zones

def refined_incident_record(df, indices, direction):
    """
    Calculates the true Peak-to-Trough (or Trough-to-Peak) for a group of indices.
    """
    start_idx = indices[0]
    end_idx = indices[-1]
    
    if direction == "DOWN":
        # Search for local peak slightly before the pattern started
        peak_search_start = max(0, start_idx - 15)
        peak_search_end = min(len(df)-1, start_idx + 5)
        peak_idx = df.iloc[peak_search_start:peak_search_end]['TA_Smooth'].idxmax()
        peak_ta = df.loc[peak_idx, 'TA']
        peak_time = df.loc[peak_idx, 'Seconds']
        
        # Search for local trough after the peak
        trough_search_start = peak_idx
        trough_search_end = min(len(df)-1, end_idx + 15)
        trough_idx = df.iloc[trough_search_start:trough_search_end]['TA_Smooth'].idxmin()
        trough_ta = df.loc[trough_idx, 'TA']
        trough_time = df.loc[trough_idx, 'Seconds']
        
        abs_change = peak_ta - trough_ta
        start_time, end_time = peak_time, trough_time
        val_start, val_end = peak_ta, trough_ta
    else: # direction == "UP"
        # Search for local trough slightly before the pattern started
        trough_search_start = max(0, start_idx - 15)
        trough_search_end = min(len(df)-1, start_idx + 5)
        trough_idx = df.iloc[trough_search_start:trough_search_end]['TA_Smooth'].idxmin()
        trough_ta = df.loc[trough_idx, 'TA']
        trough_time = df.loc[trough_idx, 'Seconds']
        
        # Search for local peak after the trough
        peak_search_start = trough_idx
        peak_search_end = min(len(df)-1, end_idx + 15)
        peak_idx = df.iloc[peak_search_start:peak_search_end]['TA_Smooth'].idxmax()
        peak_ta = df.loc[peak_idx, 'TA']
        peak_time = df.loc[peak_idx, 'Seconds']
        
        abs_change = peak_ta - trough_ta # Still Peak - Trough for magnitude
        start_time, end_time = trough_time, peak_time
        val_start, val_end = trough_ta, peak_ta

    # Calculate Change in Graph Units
    # Standard 5-Unit dial range is based on a log scale.
    log_peak = np.log10(max(0.01, peak_ta))
    log_trough = np.log10(max(0.01, trough_ta))
    
    # [LEGACY FIX] Detection thresholds based on Raw TA Drop (Ohms)
    # 0.1 TA Drop is a classic "Significant Read" for legacy data.
    # 1.5 units is the modern threshold (approx 1/3 screen).
    
    unit_change = (log_peak - log_trough) / 0.15 * 5.0
    duration = max(0.01, end_time - start_time)
    velocity = unit_change / duration
    
    # RE-CLASSIFY based on dual-thresholds (Raw TA vs Graph Units)
    new_pattern = "FALL" if direction == "DOWN" else "RISE"
    
    if direction == "DOWN":
        # BLOWDOWN: Fast or very large drop
        if velocity > 2.0 or (abs_change > 0.3 and duration < 1.2): new_pattern = "BLOWDOWN"
        # LONG FALL: Large TA drop OR high unit count
        elif abs_change >= 0.1 or unit_change >= 1.5: new_pattern = "LONG FALL"
        # SHORT FALL: Moderate drop
        elif abs_change >= 0.05 or unit_change >= 0.2: new_pattern = "SHORT FALL"
    else: # UP
        if abs(velocity) > 2.0 or (abs_change > 0.3 and duration < 1.2): new_pattern = "ROCKET READ"
        elif abs(unit_change) > 1.5 or abs_change >= 0.1: new_pattern = "LONG RISE"
        elif abs(unit_change) > 0.2 or abs_change >= 0.05: new_pattern = "SHORT RISE"
        else: new_pattern = "RISE"

    return {
        'pattern': new_pattern,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'unit_drop': abs(unit_change),
        'abs_ta_drop': abs(abs_change),
        'velocity': velocity,
        'ta_val_peak': peak_ta,
        'ta_val_trough': trough_ta,
        'val_start': val_start,
        'val_end': val_end
    }

def generate_detail_plot(df, inc):
    """Generates a zoomed-in plot of a specific incident for verification."""
    # Define window: 5s before peak to 10s after trough
    start_time = max(0, inc['start_time'] - 5)
    end_time = min(df['Seconds'].max(), inc['end_time'] + 10)
    
    mask = (df['Seconds'] >= start_time) & (df['Seconds'] <= end_time)
    detail_df = df[mask]
    
    if detail_df.empty: return None

    fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
    ax.plot(detail_df['Seconds'], detail_df['TA'], color='#004488', linewidth=1.5, label='TA')
    
    # Highlight peak and trough (start and end of movement)
    ax.scatter([inc['start_time']], [inc['val_start']], color='red', s=50, zorder=5, label='Start')
    ax.scatter([inc['end_time']], [inc['val_end']], color='green', s=50, zorder=5, label='End')
    
    # Shade the incident area
    sh_color = 'red' if 'FALL' in inc['pattern'] or 'BLOWDOWN' in inc['pattern'] else 'blue'
    ax.axvspan(inc['start_time'], inc['end_time'], color=sh_color, alpha=0.1)
    
    ax.set_title(f"Detail: {inc['pattern']} ({inc['unit_drop']:.2f} Units)")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("TA")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_wave_plot(df, wz):
    """Generates a detail plot for a Floating Wave (High-resolution detrended view)."""
    start_time = max(0, wz['start'] - 2)
    end_time = min(df['Seconds'].max(), wz['end'] + 2)
    
    mask = (df['Seconds'] >= start_time) & (df['Seconds'] <= end_time)
    detail_df = df[mask]
    
    if detail_df.empty: return None

    # We show the detrended TA to emphasize the sine wave
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=80)
    
    # Plot 1: Raw TA
    ax1.plot(detail_df['Seconds'], detail_df['TA'], color='#004488', linewidth=1.5, label='Raw TA')
    ax1.set_title(f"Floating Wave (Release Index)")
    ax1.set_ylabel("TA")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Detrended TA (Frequency View)
    if 'TA_Detrend' in detail_df.columns:
        ax2.plot(detail_df['Seconds'], detail_df['TA_Detrend'], color='#00aa00', linewidth=1.5, label='Detrended')
        ax2.axhline(y=0, color='black', alpha=0.3)
        ax2.set_ylabel("Detrended")
        ax2.set_xlabel("Seconds")
        ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_html_report(df, incidents, calib_time, session_path, wave_zones):
    """Generates a comprehensive HTML report with refined incidents."""
    
    # 1. Prepare Plot
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    fig.patch.set_facecolor('white')
    ax.plot(df['Seconds'], df['TA'], color='#004488', linewidth=0.7, label='GSR (TA Conductance)', alpha=0.9)
    
    if 'Motion' in df.columns:
        m = df[df['Motion'] == 1]
        if not m.empty: ax.scatter(m['Seconds'], m['TA'], color='#ff8800', s=4, label='Motion Lock', alpha=0.5)

    if calib_time is not None:
        ax.axvline(x=calib_time, color='red', linestyle='--', linewidth=2)
        
    # User Request: Ignore PRE-calibration data for the report
    if calib_time is not None:
        incidents = [inc for inc in incidents if inc['start_time'] >= calib_time]

    # Track top 5 events for summary
    significant_drops = sorted([inc for inc in incidents if 'FALL' in inc['pattern'] or 'BLOWDOWN' in inc['pattern']], 
                              key=lambda x: x['abs_ta_drop'], reverse=True)[:5]
    significant_rises = sorted([inc for inc in incidents if 'RISE' in inc['pattern'] or 'ROCKET' in inc['pattern']], 
                              key=lambda x: x['abs_ta_drop'], reverse=True)[:5]

    for inc in incidents:
        is_down = inc['pattern'] in ['BLOWDOWN', 'LONG FALL']
        is_up = inc['pattern'] in ['ROCKET READ', 'LONG RISE']
        color = '#cc0000' if is_down else ('#0000cc' if is_up else '#666666')
        ax.axvspan(inc['start_time'], inc['end_time'], color=color, alpha=0.15)

    ax.set_title("GSR Session Detail View (Refined Analysis)")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("TA (Conductance)")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 2. Metrics & Highlights
    total_duration = df['Seconds'].max()
    total_ta_count = df['TA'].iloc[0] - df['TA'].iloc[-1] 
    major_incidents_count = len(incidents)
    wave_event_count = len(wave_zones)

    def format_highlight(inc, idx):
        mins, secs = int(inc['start_time']) // 60, int(inc['start_time']) % 60
        detail_id = f"detail_{idx}"
        return f"""
        <div class="highlight-item" onclick="var el=document.getElementById('{detail_id}'); el.style.display='block'; el.scrollIntoView();">
            <strong>{inc['pattern']}</strong> at {mins:02}:{secs:02}<br>
            <span style="font-size:0.9em; color:#d32f2f;">{inc['abs_ta_drop']:.3f} TA</span> | {inc['unit_drop']:.1f}U
        </div>"""

    highlights_html = ""
    if significant_drops or significant_rises:
        highlights_html = f"""
        <div class="section">
            <h2>Session Audit Highlights (Top Peaks)</h2>
            <div class="highlight-grid">
                <div class="highlight-box">
                    <h3 style="color:#cc0000; border-bottom:1px solid #ffcdd2;">Top Conductance Drops</h3>
                    {"".join([format_highlight(inc, incidents.index(inc)) for inc in significant_drops]) if significant_drops else "None detected."}
                </div>
                <div class="highlight-box">
                    <h3 style="color:#0000cc; border-bottom:1px solid #bbdefb;">Top Conductance Rises</h3>
                    {"".join([format_highlight(inc, incidents.index(inc)) for inc in significant_rises]) if significant_rises else "None detected."}
                </div>
            </div>
        </div>
        """

    # 3. Build Patterns for Filter
    all_patterns = sorted(list(set([inc['pattern'] for inc in incidents])))
    filter_options = "\n".join([f'<option value="{p}">{p}</option>' for p in all_patterns])

    # 4. Tables with Detail Plots
    incident_rows = ""
    for idx, inc in enumerate(incidents):
        mins = int(inc['start_time']) // 60
        secs = int(inc['start_time']) % 60
        timestamp = f"{mins:02}:{secs:02}"
        phase = "PRE" if (calib_time is not None and inc['start_time'] < calib_time) else "POST"
        
        detail_html = ""
        # All 4 major patterns get detail plots
        detail_img_b64 = generate_detail_plot(df, inc)
        if detail_img_b64:
            detail_id = f"detail_{idx}"
            detail_html = f"""
            <button class="btn-detail" onclick="toggleDetail('{detail_id}')">View Detail</button>
            <div id="{detail_id}" class="detail-view" style="display:none;">
                <img src="data:image/png;base64,{detail_img_b64}" style="max-width:600px;">
                <p style="font-size: 0.8em; color: #666;">Verification Context: -5s before peak/trough, +10s after end.</p>
            </div>
            """
        
        row = f"""
        <tr class="incident-row" data-pattern="{inc['pattern']}">
            <td data-val="{inc['start_time']}">{timestamp} ({phase})</td>
            <td data-val="{inc['pattern']}">{inc['pattern']}</td>
            <td style="background:#fff9e6; font-weight:bold;" data-val="{inc['abs_ta_drop']}">{inc['abs_ta_drop']:.3f}</td>
            <td style="color:#666;" data-val="{inc['unit_drop']}">{inc['unit_drop']:.2f}U</td>
            <td data-val="{inc['duration']}">{inc['duration']:.2f}s</td>
            <td data-val="{abs(inc['velocity'])}">{abs(inc['velocity']):.3f} U/s</td>
            <td>{detail_html}</td>
        </tr>
        """
        incident_rows += row

    explanation_note = f"""
    <div class="explanation-box">
        <h3>Technical Audit Note: Peaks vs. Real-time</h3>
        <p>The <strong>TA Drop (Abs)</strong> represents the total change in conductance from the identified local peak to the local trough.</p>
        <p><strong>Why "Units" might seem generous:</strong> Real-time detection uses a fixed 3-second sliding window. This post-session report uses a <strong>Peak-to-Trough</strong> algorithm which identifies the <em>entire</em> logical movement, often capturing more "drop" than a real-time window would show. "Units" are provided as a secondary visual reference (estimated as inches on a 5" dial).</p>
    </div>
    """

    # 5. Floating Waves Table
    wave_rows = ""
    for idx, wz in enumerate(wave_zones):
        mins, secs = int(wz['start']) // 60, int(wz['start']) % 60
        timestamp = f"{mins:02}:{secs:02}"
        duration = wz['end'] - wz['start']
        
        detail_id = f"wave_{idx}"
        wave_img = generate_wave_plot(df, wz)
        detail_html = ""
        if wave_img:
            detail_html = f"""
            <button class="btn-detail" style="background:#2e7d32;" onclick="toggleDetail('{detail_id}')">View Waveform</button>
            <div id="{detail_id}" class="detail-view" style="display:none; border-color:#2e7d32;">
                <img src="data:image/png;base64,{wave_img}" style="max-width:600px;">
                <p style="font-size: 0.8em; color: #2e7d32;">Floating Wave: Rhythmic Release Pattern ({duration:.1f}s duration).</p>
            </div>
            """
            
        wave_rows += f"""
        <tr>
            <td data-val="{wz['start']}">{timestamp}</td>
            <td>Floating Wave</td>
            <td data-val="{duration}">{duration:.1f}s</td>
            <td>{detail_html}</td>
        </tr>
        """
    
    waves_section = ""
    if wave_rows:
        waves_section = f"""
        <div class="section">
            <h2 style="color:#2e7d32;">Floating Wave Analysis (Release Indicator)</h2>
            <p style="font-size: 0.9em; color: #555;">Detected rhythmic sine-wave oscillations indicating "Release" or "End Phenomenon".</p>
            <table>
                <thead><tr><th onclick="sortTable(0, this)">Time ⇳</th><th>Type</th><th onclick="sortTable(2, this)">Duration ⇳</th><th>Detail</th></tr></thead>
                <tbody>{wave_rows}</tbody>
            </table>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>GSR Refined Analysis Report - {os.path.basename(session_path)}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f7f6; }}
            .header {{ background: #004488; color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }}
            .section {{ background: white; padding: 25px; margin-bottom: 20px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
            .highlight-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .highlight-box {{ background: #fff; padding: 15px; border: 1px solid #ddd; border-radius: 6px; }}
            .highlight-item {{ padding: 8px; margin-bottom: 5px; background: #fafafa; border-radius: 4px; cursor: pointer; transition: background 0.2s; }}
            .highlight-item:hover {{ background: #f0f0f0; }}
            .metric-card {{ background: #eef4fb; padding: 15px; border-radius: 6px; border-left: 5px solid #004488; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #004488; }}
            img {{ width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; cursor: pointer; }}
            th:hover {{ background-color: #e9ecef; }}
            .filter-container {{ margin-bottom: 15px; background: #eee; padding: 15px; border-radius: 4px; }}
            .btn-detail {{ padding: 4px 8px; cursor: pointer; background: #004488; color: white; border: none; border-radius: 4px; }}
            .detail-view {{ margin-top: 10px; border: 1px solid #ccc; padding: 10px; background: #fff; }}
            .explanation-box {{ background: #fffde7; border-left: 5px solid #fbc02d; padding: 15px; margin-top: 20px; font-size: 0.9em; }}
        </style>
        <script>
            function filterPatterns() {{
                const val = document.getElementById('patternFilter').value;
                const rows = document.querySelectorAll('.incident-row');
                rows.forEach(row => {{
                    if (val === 'ALL' || row.dataset.pattern === val) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }});
            }}
            function toggleDetail(id) {{
                const el = document.getElementById(id);
                el.style.display = (el.style.display === 'none') ? 'block' : 'none';
            }}
            function sortTable(n, header) {{
                let table = header.closest("table");
                let rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                switching = true;
                dir = "asc";
                while (switching) {{
                    switching = false;
                    rows = table.rows;
                    for (i = 1; i < (rows.length - 1); i++) {{
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName("TD")[n];
                        y = rows[i + 1].getElementsByTagName("TD")[n];
                        
                        // Use data-val if available, otherwise innerText
                        let xVal = x.dataset.val ? parseFloat(x.dataset.val) : x.innerText.toLowerCase();
                        let yVal = y.dataset.val ? parseFloat(y.dataset.val) : y.innerText.toLowerCase();
                        
                        if (typeof xVal === 'string') {{
                             if (dir == "asc") {{ if (xVal > yVal) {{ shouldSwitch = true; break; }} }}
                             else if (dir == "desc") {{ if (xVal < yVal) {{ shouldSwitch = true; break; }} }}
                        }} else {{
                             if (dir == "asc") {{ if (xVal > yVal) {{ shouldSwitch = true; break; }} }}
                             else if (dir == "desc") {{ if (xVal < yVal) {{ shouldSwitch = true; break; }} }}
                        }}
                    }}
                    if (shouldSwitch) {{
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount ++;
                    }} else {{
                        if (switchcount == 0 && dir == "asc") {{
                            dir = "desc";
                            switching = true;
                        }}
                    }}
                }}
            }}
        </script>
    </head>
    <body>
        <div class="header"><h1>GSR Refined Session Report</h1><p>{os.path.basename(session_path)}</p></div>
        <div class="section">
            <h2>Refined Summary</h2>
            <div class="summary-grid">
                <div class="metric-card"><div class="metric-value">{int(total_duration // 60)}:{int(total_duration % 60):02}</div><div class="metric-label">Duration</div></div>
                <div class="metric-card"><div class="metric-value">{total_ta_count:.3f}</div><div class="metric-label">Net TA Change</div></div>
                <div class="metric-card"><div class="metric-value">{major_incidents_count}</div><div class="metric-label">Significant Incidents</div></div>
                <div class="metric-card"><div class="metric-value">{wave_event_count}</div><div class="metric-label">Floating Waves (Release)</div></div>
            </div>
        </div>
        {highlights_html}
        {waves_section}
        <div class="section"><h2>GSR Peak-to-Trough Analysis Graph</h2><img src="data:image/png;base64,{img_b64}"></div>
        <div class="section">
            <h2>Incident log (Audit View - Click Headers to Sort)</h2>
            <div class="filter-container">
                <label for="patternFilter">Filter by Pattern: </label>
                <select id="patternFilter" onchange="filterPatterns()">
                    <option value="ALL">-- All Patterns --</option>
                    {filter_options}
                </select>
            </div>
            <table>
                <thead><tr>
                    <th onclick="sortTable(0, this)">Time (Phase) ⇳</th>
                    <th onclick="sortTable(1, this)">Type ⇳</th>
                    <th onclick="sortTable(2, this)">TA Change (Abs) ⇳</th>
                    <th onclick="sortTable(3, this)">Est. Units ⇳</th>
                    <th onclick="sortTable(4, this)">Duration ⇳</th>
                    <th onclick="sortTable(5, this)">Velocity ⇳</th>
                    <th>Detail</th>
                </tr></thead>
                <tbody>{incident_rows if incident_rows else '<tr><td colspan="7">No major incidents (LONG FALL, BLOWDOWN, LONG RISE, ROCKET READ) detected.</td></tr>'}</tbody>
            </table>
            {explanation_note}
        </div>
        <div class="footer" style="text-align:center; padding:20px; color:#888;">
            * Analysis filtered for significance: LONG FALL/RISE > 1.5 Units. BLOWDOWN/ROCKET > 2.0 U/s.
        </div>
    </body>
    </html>
    """
    
    #report_path = os.path.abspath("session_report.html")
    for path in [os.path.join(session_path, "session_report.html")]:
        with open(path, "w") as f: f.write(html_content)
    
    print(f"Refined Report generated at: {session_path}")

def plot_session(df, session_path):
    """Generates the high-detail GSR analysis plot with refined annotations."""
    calib_time = None
    if 'Notes' in df.columns:
        # Priority 1: CALIB_COMPLETE or CALIB_END
        calib_ends = df[df['Notes'].str.contains("CALIB_COMPLETE|CALIB_END", na=False)]
        if not calib_ends.empty:
            calib_time = calib_ends['Seconds'].iloc[0]
        else:
            # Priority 2: CALIB_START (Fallback)
            calib_pts = df[df['Notes'].str.contains("CALIB_START", na=False)]
            if not calib_pts.empty: calib_time = calib_pts['Seconds'].iloc[0]
            
    # Professional Analytics
    incidents, wave_zones = analyze_incidents(df)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.plot(df['Seconds'], df['TA'], color='#004488', linewidth=0.7, alpha=0.9)
     
    # Mark Floating Waves (Release Phenomena)
    for wz in wave_zones:
        ax.axvspan(wz['start'], wz['end'], color='#00ff00', alpha=0.1, label='Floating Wave' if wz == wave_zones[0] else "")
        ax.text(wz['start'], ax.get_ylim()[0], ' WAVE ', color='green', fontsize=7, rotation=90)

    if 'Motion' in df.columns:
        m = df[df['Motion'] == 1]
        if not m.empty: ax.scatter(m['Seconds'], m['TA'], color='#ff8800', s=4, alpha=0.5)

    if calib_time is not None:
        ax.axvline(x=calib_time, color='red', linestyle='--')
        ax.text(calib_time, ax.get_ylim()[1], ' PRE | POST ', color='red', fontweight='bold', ha='center')

    for inc in incidents:
        is_down = inc['pattern'] in ['BLOWDOWN', 'LONG FALL']
        is_up = inc['pattern'] in ['ROCKET READ', 'LONG RISE']
        color = '#cc0000' if is_down else ('#0000cc' if is_up else '#666666')
        ax.axvspan(inc['start_time'], inc['end_time'], color=color, alpha=0.15)
        
        label = f"{inc['pattern']}\n{inc['unit_drop']:.1f}U"
        ax.text(inc['start_time'], inc['val_start'], label, color=color, fontsize=8, rotation=45, fontweight='bold')

    plt.savefig(os.path.join(session_path, "analysis_result.png"))
    #plt.savefig("latest_analysis.png")
    plt.close(fig)
    generate_html_report(df, incidents, calib_time, session_path, wave_zones)

if __name__ == "__main__":
    selected_dir = select_session()
    if selected_dir:
        print(f"Analyzing: {selected_dir}")
        df = load_data(selected_dir)
        if df is not None: plot_session(df, selected_dir)
