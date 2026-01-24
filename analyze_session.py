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

def detect_calibration_time(df):
    """Finds the moment session started (after calibration)."""
    if 'Notes' not in df.columns:
        return None
    # Priority 1: CALIB_COMPLETE or CALIB_END
    mask = df['Notes'].str.contains("CALIB_COMPLETE|CALIB_END", na=False)
    if mask.any():
        return df[mask]['Seconds'].iloc[0]
    # Priority 2: CALIB_START (Fallback)
    mask = df['Notes'].str.contains("CALIB_START", na=False)
    if mask.any():
        return df[mask]['Seconds'].iloc[0]
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

def generate_session_synopsis(duration, net_ta, incidents, wave_count, question_analysis):
    """
    Generates a narrative session overview in the voice of a professional session reviewer.
    """
    # 1. Determine Overall Tone
    tone = ""
    net_ta_val = float(net_ta)
    
    if net_ta_val < -0.5:
        tone = "The session demonstrates a significant overall release of tension, indicating a successful unburdening process."
    elif net_ta_val > 0.5:
        tone = "The session shows a trend of accumulating tension, suggesting the participant encountered difficult material that remained largely unresolved."
    else:
        tone = "The session remained relatively balanced, with minor fluctuations in tension but no dramatic overall shift."
        
    activity_level = ""
    if len(incidents) > 15:
        activity_level = "Reactivity was high, suggesting a volatile session with frequent emotional responses."
    elif len(incidents) < 5:
        activity_level = "Reactivity was notably low, indicating a calm or perhaps defended session."
    else:
        activity_level = "Reactivity was moderate, consistent with a standard processing session."
        
    flow_desc = ""
    if wave_count > 3:
        flow_desc = "The presence of multiple floating waves suggests deep, rhythmic processing and a good flow of release."
    elif wave_count > 0:
        flow_desc = "Some rhythmic processing was observed, indicating moments of flow."
    else:
        flow_desc = "The absence of floating waves suggests the session may have been more cognitive or rigid, lacking deep rhythmic release."

    intro_paragraph = f"<p><strong>General Synopsis:</strong> This {int(duration/60)} minute session presented a distinct energetic profile. {tone} {activity_level} {flow_desc}</p>"

    # 2. Key Observations
    observations = []
    
    # Biggest Release
    drops = [i for i in incidents if i['unit_drop'] > 1.0]
    if drops:
        max_drop = max(drops, key=lambda x: x['unit_drop'])
        observations.append(f"<li><strong>Major Release:</strong> A significant release of {max_drop['unit_drop']:.1f} TA Units was observed at {int(max_drop['start_time']//60)}:{int(max_drop['start_time']%60):02}, characterized as a {max_drop['pattern']}.</li>")
        
    # Question Reactivity
    if question_analysis:
        # Most reactive question (Net Proc TA magnitude)
        most_reactive = max(question_analysis, key=lambda x: abs(x['stats']['net_proc_ta']))
        q_text = most_reactive['question']['text']
        q_change = most_reactive['stats']['net_proc_ta']
        change_type = "drop" if q_change < 0 else "rise"
        observations.append(f"<li><strong>Highest Reactivity:</strong> The question '{q_text}' elicited the strongest response, resulting in a net {change_type} of {abs(q_change):.3f} TA.</li>")
        
    # Wave Pattern
    if wave_count > 0:
        observations.append(f"<li><strong>Rhythmic Flow:</strong> {wave_count} 'Floating Wave' events were detected, marking periods of sustained, rhythmic processing.</li>")

    obs_list = "<ul>" + "".join(observations) + "</ul>" if observations else ""
    
    return f"{intro_paragraph}<h4>Key Observations</h4>{obs_list}"

def generate_html_report(df, incidents, calib_time, session_path, wave_zones, question_analysis=None):
    """Generates a comprehensive HTML report using an external template."""
    
    # 1. Prepare Main Plot
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    fig.patch.set_facecolor('white')
    ax.plot(df['Seconds'], df['TA'], color='#004488', linewidth=0.7, label='GSR (TA Conductance)', alpha=0.9)
    
    if 'Motion' in df.columns:
        m = df[df['Motion'] == 1]
        if not m.empty: ax.scatter(m['Seconds'], m['TA'], color='#ff8800', s=4, label='Motion Lock', alpha=0.5)

    if calib_time is not None:
        ax.axvline(x=calib_time, color='red', linestyle='--', linewidth=2)
        
    # User Request: Ignore PRE-calibration data for the report incidents
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
    # Safely handle empty data
    if len(df) > 0:
        total_ta_count = df['TA'].iloc[0] - df['TA'].iloc[-1] 
    else:
        total_ta_count = 0
    major_incidents_count = len(incidents)
    wave_event_count = len(wave_zones)

    # 2.5 Associate patterns with questions [NEW]
    if question_analysis:
        for qa in question_analysis:
            q_start = qa['question']['time']
            q_end = q_start + qa['stats']['duration_proc']
            # Find all incidents that started within this question's window
            qa['incidents'] = [inc for inc in incidents if q_start <= inc['start_time'] < q_end]
            
    def format_highlight(inc, idx):
        mins, secs = int(inc['start_time']) // 60, int(inc['start_time']) % 60
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
    # Safely handle empty data
    if len(df) > 0:
        total_ta_count = df['TA'].iloc[0] - df['TA'].iloc[-1] 
    else:
        total_ta_count = 0
    major_incidents_count = len(incidents)
    wave_event_count = len(wave_zones)

    # 2.5 Associate patterns with questions [NEW]
    if question_analysis:
        for qa in question_analysis:
            q_start = qa['question']['time']
            q_end = q_start + qa['stats']['duration_proc']
            # Find all incidents that started within this question's window
            qa['incidents'] = [inc for inc in incidents if q_start <= inc['start_time'] < q_end]
            
    def format_highlight(inc, idx):
        mins, secs = int(inc['start_time']) // 60, int(inc['start_time']) % 60
        detail_id = f"detail_{idx}"
        return f"""
        <div class="highlight-item" onclick="var el=document.getElementById('{detail_id}'); el.style.display='block'; el.scrollIntoView();">
            <strong>{inc['pattern']}</strong> at {mins:02}:{secs:02}<br>
            <span style="font-size:0.9em; color:#d32f2f;">{inc['abs_ta_drop']:.3f} TA</span> | {inc['unit_drop']:.1f}U
        </div>"""

    def format_q_highlight(qa, label, val, color):
        q_data = qa['question']
        mins, secs = int(q_data['time']) // 60, int(q_data['time']) % 60
        detail_id = f"q_detail_{question_analysis.index(qa)}"
        return f"""
        <div class="highlight-item" onclick="var el=document.getElementById('{detail_id}'); el.style.display='block'; el.scrollIntoView();">
            <span style="color:gray; font-size:0.75em;">{mins:02}:{secs:02} {q_data['marker']}</span><br>
            <strong>{label}:</strong> <span style="color:{color};">{val:.3f}</span><br>
            <span style="font-size:0.8em; font-style:italic;">{q_data['text'][:40]}...</span>
        </div>"""

    highlights_html = ""
    # 1. Incident Highlights
    if significant_drops or significant_rises:
        highlights_html += f"""
        <div class="section">
            <h2>Peak Pattern Highlights (Incident Log)</h2>
            <div class="highlight-grid">
                <div class="highlight-box" style="border-top: 5px solid #cc0000;">
                    <h3 style="color:#cc0000; margin-top:0;">Top Patterns: Drops</h3>
                    {"".join([format_highlight(inc, incidents.index(inc)) for inc in significant_drops]) if significant_drops else "None detected."}
                </div>
                <div class="highlight-box" style="border-top: 5px solid #0000cc;">
                    <h3 style="color:#0000cc; margin-top:0;">Top Patterns: Rises</h3>
                    {"".join([format_highlight(inc, incidents.index(inc)) for inc in significant_rises]) if significant_rises else "None detected."}
                </div>
            </div>
        </div>
        """
    
    # 2. Question Highlights [NEW: Triple View - 15s, Net Proc, Max Pattern]
    if question_analysis:
        # 1. Top 15s Response (Immediate)
        top_15_drop = sorted(question_analysis, key=lambda x: x['stats']['drop_15'])[:3]
        top_15_rise = sorted(question_analysis, key=lambda x: x['stats']['rise_15'], reverse=True)[:3]
        
        # 2. Top Net Processing (Cumulative)
        top_net_drop = sorted(question_analysis, key=lambda x: x['stats']['net_proc_ta'])[:3]
        top_net_rise = sorted(question_analysis, key=lambda x: x['stats']['net_proc_ta'], reverse=True)[:3]

        # 3. Top Cumulative Totals (Activity) [NEW]
        top_total_drop = sorted(question_analysis, key=lambda x: x['stats']['total_drop_proc'])[:3]
        top_total_rise = sorted(question_analysis, key=lambda x: x['stats']['total_rise_proc'], reverse=True)[:3]
        
        highlights_html += f"""
        <div class="section">
            <h2 style="margin-bottom:10px;">Question Audit Highlights</h2>
            
            <div style="margin-bottom:20px;">
                <h3 style="color:#d4af37; border-bottom: 2px solid gold; margin-bottom:10px;">Top Initial Response (15s Window)</h3>
                <div class="highlight-grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="highlight-box" style="border-top: 5px solid #cc0000; background: #fff5f5;">
                        <h4 style="color:#cc0000; margin-top:0;">Biggest Initial Drop</h4>
                        {"".join([format_q_highlight(qa, "15s Drop", qa['stats']['drop_15'], "#cc0000") for qa in top_15_drop if qa['stats']['drop_15'] < -0.01]) or "None."}
                    </div>
                    <div class="highlight-box" style="border-top: 5px solid #00aa00; background: #f5fff5;">
                        <h4 style="color:#00aa00; margin-top:0;">Biggest Initial Rise</h4>
                        {"".join([format_q_highlight(qa, "15s Rise", qa['stats']['rise_15'], "#00aa00") for qa in top_15_rise if qa['stats']['rise_15'] > 0.01]) or "None."}
                    </div>
                </div>
            </div>

            <div style="margin-bottom:20px;">
                <h3 style="color:#2e7d32; border-bottom: 2px solid #2e7d32; margin-bottom:10px;">Processing Trends (Net Shift)</h3>
                <div class="highlight-grid" style="grid-template-columns: repeat(2, 1fr);">
                    <div class="highlight-box" style="border-top: 5px solid #cc0000; background: #fff5f5;">
                        <h4 style="color:#cc0000; margin-top:0;">Max Net Drop (Overall Shift)</h4>
                        {"".join([format_q_highlight(qa, "Net", qa['stats']['net_proc_ta'], "#cc0000") for qa in top_net_drop if qa['stats']['net_proc_ta'] < -0.01]) or "None."}
                    </div>
                    <div class="highlight-box" style="border-top: 5px solid #00aa00; background: #f5fff5;">
                        <h4 style="color:#00aa00; margin-top:0;">Max Net Rise (Overall Shift)</h4>
                        {"".join([format_q_highlight(qa, "Net", qa['stats']['net_proc_ta'], "#00aa00") for qa in top_net_rise if qa['stats']['net_proc_ta'] > 0.01]) or "None."}
                    </div>
                </div>
            </div>

            <div>
                <h3 style="color:#666; border-bottom: 2px solid #666; margin-bottom:10px;">Processing Movement (Cumulative Totals)</h3>
                <div class="highlight-grid" style="grid-template-columns: repeat(2, 1fr);">
                    <div class="highlight-box" style="border-top: 5px solid #cc0000; background: #fff5f5;">
                        <h4 style="color:#cc0000; margin-top:0;">Max Total Drop (Activity)</h4>
                        {"".join([format_q_highlight(qa, "Sum Drop", qa['stats']['total_drop_proc'], "#cc0000") for qa in top_total_drop if qa['stats']['total_drop_proc'] < -0.01]) or "None."}
                    </div>
                    <div class="highlight-box" style="border-top: 5px solid #0000cc; background: #f5f5ff;">
                        <h4 style="color:#0000cc; margin-top:0;">Max Total Rise (Activity)</h4>
                        {"".join([format_q_highlight(qa, "Sum Rise", qa['stats']['total_rise_proc'], "#0000cc") for qa in top_total_rise if qa['stats']['total_rise_proc'] > 0.01]) or "None."}
                    </div>
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
        
        # Find Question Context [NEW]
        q_context = "N/A"
        if question_analysis:
            for qa in question_analysis:
                q_start = qa['question']['time']
                q_end = q_start + qa['stats']['duration_proc']
                if q_start <= inc['start_time'] < q_end:
                    q_context = qa['question']['marker']
                    break

        detail_html = ""
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
            <td>{q_context}</td>
            <td data-val="{inc['pattern']}">{inc['pattern']}</td>
            <td style="background:#fff9e6; font-weight:bold;" data-val="{inc['abs_ta_drop']}">{inc['abs_ta_drop']:.3f}</td>
            <td style="color:#666;" data-val="{inc['unit_drop']}">{inc['unit_drop']:.2f}U</td>
            <td data-val="{inc['duration']}">{inc['duration']:.2f}s</td>
            <td data-val="{abs(inc['velocity'])}">{abs(inc['velocity']):.3f} U/s</td>
            <td>{detail_html}</td>
        </tr>
        """
        incident_rows += row

    # 5. Question Analysis Section [NEW]
    question_html = ""
    if question_analysis:
        q_rows = ""
        for idx, qa in enumerate(question_analysis):
            q_data = qa['question']
            st = qa['stats']
            mins, secs = int(q_data['time']) // 60, int(q_data['time']) % 60
            timestamp = f"{mins:02}:{secs:02}"
            
            detail_id = f"q_detail_{idx}"
            q_img = generate_question_plot(df, q_data, st)
            
            # Find largest pattern for display
            max_p_drop = 0
            max_p_rise = 0
            if qa.get('incidents'):
                drops = [inc['abs_ta_drop'] for inc in qa.get('incidents', []) if 'FALL' in inc['pattern'] or 'BLOWDOWN' in inc['pattern']]
                rises = [inc['abs_ta_drop'] for inc in qa.get('incidents', []) if 'RISE' in inc['pattern'] or 'ROCKET' in inc['pattern']]
                max_p_drop = max(drops) if drops else 0
                max_p_rise = max(rises) if rises else 0
            
            q_rows += f'''
            <div id="q_detail_{idx}" class="highlight-box" style="margin-bottom:20px; border-left: 5px solid gold;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="flex:1;">
                        <span style="color:gray; font-size:0.8em;">{timestamp} ({q_data['marker']})</span>
                        <h3 style="margin:5px 0;">{q_data['text']}</h3>
                        <div class="summary-grid" style="grid-template-columns: repeat(5, 1fr); gap:10px; margin-top:10px;">
                            <div class="metric-card" style="padding:10px; border-left-color: gold;">
                                <div style="font-size:0.8em; color:gray;">15s Response</div>
                                <div style="font-size:1.0em; font-weight:bold;">{st['drop_15']:.3f} / +{st['rise_15']:.3f}</div>
                            </div>
                            <div class="metric-card" style="padding:10px; border-left-color: #cc0000;">
                                <div style="font-size:0.8em; color:gray;">Proc Total Drop</div>
                                <div style="font-size:1.0em; font-weight:bold;">{st['total_drop_proc']:.3f}</div>
                            </div>
                            <div class="metric-card" style="padding:10px; border-left-color: #0000cc;">
                                <div style="font-size:0.8em; color:gray;">Proc Total Rise</div>
                                <div style="font-size:1.0em; font-weight:bold;">+{st['total_rise_proc']:.3f}</div>
                            </div>
                            <div class="metric-card" style="padding:10px; border-left-color: #2e7d32; background: {'#fff9f9' if st['net_proc_ta'] < 0 else '#f9fff9'};">
                                <div style="font-size:0.8em; color:gray;">Proc Net TA</div>
                                <div style="font-size:1.0em; font-weight:bold; color: {'#cc0000' if st['net_proc_ta'] < 0 else '#00aa00'};">{st['net_proc_ta']:+.3f}</div>
                            </div>
                            <div class="metric-card" style="padding:10px; border-left-color: gray;">
                                <div style="font-size:0.8em; color:gray;">Proc Time</div>
                                <div style="font-size:1.0em; font-weight:bold;">{st['duration_proc']:.1f}s</div>
                            </div>
                        </div>
                    </div>
                    <div style="width:400px; margin-left:20px;">
                        <img src="data:image/png;base64,{q_img}" style="width:100%; cursor:pointer;" onclick="toggleDetail('{detail_id}_full')">
                    </div>
                </div>
                <div id="{detail_id}_full" class="detail-view" style="display:none; text-align:center;">
                     <img src="data:image/png;base64,{q_img}" style="max-width:800px;">
                </div>
                
            '''
            if qa.get('incidents'):
                inc_html = []
                for i, inc in enumerate(qa["incidents"]):
                        q_inc_id = f"q_{idx}_inc_{i}"
                        # Reuse existing detail generator
                        d_img = generate_detail_plot(df, inc)
                        btn = ""
                        div = ""
                        if d_img:
                            btn = f'<button class="btn-detail" style="font-size:0.75em; padding:2px 5px; margin-left:10px;" onclick="toggleDetail(\'{q_inc_id}\')">View Detail</button>'
                            div = f'<div id="{q_inc_id}" class="detail-view" style="display:none;"><img src="data:image/png;base64,{d_img}" style="max-width:400px;"></div>'
                        
                        inc_html.append(f'<div><b>{inc["pattern"]}:</b> {inc["abs_ta_drop"]:.3f} TA ({inc["unit_drop"]:.1f} Units){btn}{div}</div>')
                
                q_rows += f'''
                <div style="margin-top:10px; background:#f0f0f0; padding:10px; border-radius:4px;">
                    <h5 style="margin:0 0 5px 0; color:#555;">Detected Patterns in this Window:</h5>
                    {"".join(inc_html)}
                </div>
                ''' 
            
            q_rows += "</div>" # Close the main question box div
            
        question_html = "<div class=\"section\">"
        question_html += "<h2 style=\"color:#d4af37; border-bottom: 2px solid gold; padding-bottom:5px;\">Question-by-Question Analysis</h2>"
        question_html += "<p style=\"font-size: 0.9em; color: gray;\">Detailed response tracking from moment of delivery (15s window & total processing time until next Q).</p>"
        question_html += q_rows
        question_html += "</div>"

    explanation_note = ""
    explanation_note += "<div class=\"explanation-box\">"
    explanation_note += "<h3>Technical Audit Note: Peaks vs. Cumulative Totals</h3>"
    explanation_note += "<p><strong>Standardization:</strong> All TA changes follow mathematical deltas (End - Start). Rises are <strong>Positive (+)</strong> and Drops are <strong>Negative (-)</strong>.</p>"
    explanation_note += "<p><strong>Why is \"Total Rise\" often larger than any single peak?</strong> \"Proc Total Rise\" is the <em>sum of every upward movement</em> during the processing period. For example, if the client rises 0.1, drops, and rises 0.1 again, the Total Rise is 0.2, even though the max single rise was only 0.1.</p>"
    explanation_note += "</div>"

    wave_rows = ""
    for idx, wz in enumerate(wave_zones):
        mins, secs = int(wz['start']) // 60, int(wz['start']) % 60
        timestamp = f"{mins:02}:{secs:02}"
        duration = wz['end'] - wz['start']
        
        detail_id = f"wave_{idx}"
        wave_img = generate_wave_plot(df, wz)
        detail_html = ""
        if wave_img:
            detail_html = f'''
            <button class="btn-detail" style="background:#2e7d32;" onclick="toggleDetail('{detail_id}')">View Waveform</button>
            <div id="{detail_id}" class="detail-view" style="display:none; border-color:#2e7d32;">
                <img src="data:image/png;base64,{wave_img}" style="max-width:600px;">
                <p style="font-size: 0.8em; color: #2e7d32;">Floating Wave: Rhythmic Release Pattern ({duration:.1f}s duration).</p>
            </div>
            '''
            
        dur_str = f"{duration:.1f}"
        wave_rows += "<tr>"
        wave_rows += "<td data-val=\"" + str(wz['start']) + "\">" + timestamp + "</td>"
        wave_rows += "<td>Floating Wave</td>"
        wave_rows += "<td data-val=\"" + str(duration) + "\">" + dur_str + "s</td>"
        wave_rows += "<td>" + detail_html + "</td>"
        wave_rows += "</tr>"
    
    waves_section = ""
    if wave_rows:
        waves_section = """
        <div class="section">
            <h2 style="color:#2e7d32;">Floating Wave Analysis (Release Indicator)</h2>
            <p style="font-size: 0.9em; color: dimgray;">Detected rhythmic sine-wave oscillations indicating "Release" or "End Phenomenon".</p>
            <table>
                <thead><tr><th onclick="sortTable(0, this)">Time &#8691;</th><th>Type</th><th onclick="sortTable(2, this)">Duration &#8691;</th><th>Detail</th></tr></thead>
                <tbody>{}</tbody>
            </table>
            <div class="explanation-box" style="border-left: 5px solid #2e7d32;">
                <strong>Metric Note:</strong> Floating waves are distinctly different from typical falls. They represent a state of 'undulating release', often seen when a subject is processing deep relief or detaching from a charge.
            </div>
        </div>
        """.format(wave_rows)

    # Load Template
    try:
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as e:
        print(f"Error loading report template: {e}")
        return

    # Prepare Data for Injection
    session_name = os.path.basename(session_path)
    duration_str = f"{int(total_duration // 60)}:{int(total_duration % 60):02}"
    net_ta_str = f"{total_ta_count:.3f}"
    
    # 1. Inject Simple Metrics
    html_content = html_content.replace("{{ SESSION_NAME }}", session_name)
    html_content = html_content.replace("{{ DURATION }}", duration_str)
    html_content = html_content.replace("{{ NET_TA_CHANGE }}", net_ta_str)
    html_content = html_content.replace("{{ INCIDENT_COUNT }}", str(major_incidents_count))
    html_content = html_content.replace("{{ WAVE_COUNT }}", str(wave_event_count))
    
    # 2. Inject Sections
    # 2. Inject Sections
    synopsis_html = generate_session_synopsis(total_duration, total_ta_count, incidents, wave_event_count, question_analysis)
    html_content = html_content.replace("{{ SESSION_OVERVIEW }}", synopsis_html) 
    
    html_content = html_content.replace("{{ HIGHLIGHTS_SECTION }}", highlights_html)
    html_content = html_content.replace("{{ QUESTION_SECTION }}", question_html)
    html_content = html_content.replace("{{ WAVES_SECTION }}", waves_section)
    html_content = html_content.replace("{{ MAIN_IMAGE_B64 }}", img_b64)
    
    # 3. Inject Tables & Filters
    html_content = html_content.replace("{{ FILTER_OPTIONS }}", filter_options)
    html_content = html_content.replace("{{ INCIDENT_ROWS }}", incident_rows if incident_rows else '<tr><td colspan="8">No major incidents detected.</td></tr>')
    html_content = html_content.replace("{{ EXPLANATION_NOTE }}", explanation_note)
    
    # Write Final Report
    for path in [os.path.join(session_path, "session_report.html")]:
        with open(path, "w", encoding="utf-8") as f: f.write(html_content)
    
    print(f"Refined Report generated at: {session_path}")

def extract_questions(df):
    """Identifies all question start points and their associated text."""
    questions = []
    if 'Notes' not in df.columns:
        return questions

    # Filter for _START markers in Notes
    qs_mask = df['Notes'].notna() & df['Notes'].str.contains('_START', na=False)
    qs_transitions = df[qs_mask]
    
    processed_notes = set()
    for _, row in qs_transitions.iterrows():
        note = str(row['Notes'])
        if note not in processed_notes:
            processed_notes.add(note)
            q_text = row.get('Question_Text', note)
            if pd.isna(q_text) or str(q_text).strip() == "":
                q_text = note
            
            questions.append({
                'time': float(row['Seconds']),
                'marker': note,
                'text': str(q_text).strip()
            })
    return questions

def analyze_question_response(df, q_data, next_q_time=None):
    """Calculates metrics for a specific question window."""
    start_time = q_data['time']
    # 15s Window
    win_15_end = min(df['Seconds'].max(), start_time + 15)
    mask_15 = (df['Seconds'] >= start_time) & (df['Seconds'] <= win_15_end)
    view_15 = df[mask_15]
    
    # Full Processing Window
    proc_end = next_q_time if next_q_time else df['Seconds'].max()
    mask_proc = (df['Seconds'] >= start_time) & (df['Seconds'] <= proc_end)
    view_proc = df[mask_proc]
    
    # [NEW] Cumulative TA Movement Analysis with Noise Floor
    if not view_proc.empty:
        # Calculate differences between consecutive points
        diffs = view_proc['TA'].diff().dropna()
        
        # Noise Floor (0.0001): lowered to capture gradual movements [FIX]
        # This ensures that even slow 0.14 rises aren't filtered out by ADC-jitter-protection
        noise_floor = 0.0001
        
        # Total Drop: sum of all decreases (negative diffs)
        total_drop = diffs[diffs < -noise_floor].sum()
        # Total Rise: sum of all increases (positive diffs)
        total_rise = diffs[diffs > noise_floor].sum()
        
        # Net Change: Final - Initial (Positive if Rise, Negative if Drop)
        net_proc_ta = view_proc['TA'].iloc[-1] - view_proc['TA'].iloc[0]
    else:
        total_drop = total_rise = net_proc_ta = 0

    stats = {
        'start_ta': view_15['TA'].iloc[0] if not view_15.empty else 0,
        'min_15_ta': view_15['TA'].min() if not view_15.empty else 0,
        'max_15_ta': view_15['TA'].max() if not view_15.empty else 0,
        'net_proc_ta': net_proc_ta,
        'total_drop_proc': total_drop,
        'total_rise_proc': total_rise,
        'duration_proc': (proc_end - start_time)
    }
    
    # Standardize 15s Window signs (End-Start logic for deltas)
    if not view_15.empty:
        # Drop is the delta from start to the lowest point (will be negative)
        stats['drop_15'] = view_15['TA'].min() - stats['start_ta']
        # Rise is the delta from start to the highest point (will be positive)
        stats['rise_15'] = view_15['TA'].max() - stats['start_ta']
    else:
        stats['drop_15'] = stats['rise_15'] = 0
        
    return stats

def generate_question_plot(df, q_data, stats):
    """Generates a graph for the full processing interval (until next question)."""
    start_time = q_data['time']
    end_time = min(df['Seconds'].max(), start_time + stats['duration_proc'])
    
    mask = (df['Seconds'] >= start_time - 2) & (df['Seconds'] <= end_time + 2)
    detail_df = df[mask]
    
    if detail_df.empty: return None

    fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
    ax.plot(detail_df['Seconds'], detail_df['TA'], color='#004488', linewidth=1.5)
    
    # Highlight the initial 15s window specifically within the larger plot
    win_15_end = min(df['Seconds'].max(), start_time + 15)
    ax.axvspan(start_time, win_15_end, color='gold', alpha=0.1, label='15s Response')
    # Full processing window background
    ax.axvspan(start_time, end_time, color='gray', alpha=0.05)
    ax.axvline(x=start_time, color='gold', linestyle='--', alpha=0.8, label='Q Asked')
    
    ax.set_title(f"Q: {q_data['text'][:50]}...")
    ax.set_ylabel("TA")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_session(df, session_path):
    """Generates the high-detail GSR analysis plot with refined annotations."""
    calib_time = detect_calibration_time(df)
            
    # Professional Analytics
    incidents, wave_zones = analyze_incidents(df)
    
    # Filter by calibration time [REQ]
    if calib_time is not None:
        incidents = [inc for inc in incidents if inc['start_time'] >= calib_time]
        wave_zones = [wz for wz in wave_zones if wz['start'] >= calib_time]

    # [NEW] Question-by-Question Analysis
    questions = extract_questions(df)
    # Filter questions by calibration time [REQ]
    if calib_time is not None:
        questions = [q for q in questions if q['time'] >= calib_time]
        
    question_analysis = []
    for i, q in enumerate(questions):
        next_t = questions[i+1]['time'] if i+1 < len(questions) else None
        res_stats = analyze_question_response(df, q, next_t)
        question_analysis.append({
            'question': q,
            'stats': res_stats
        })

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.plot(df['Seconds'], df['TA'], color='#004488', linewidth=0.7, alpha=0.9)
     
    # Mark Floating Waves (Release Phenomena)
    for wz in wave_zones:
        ax.axvspan(wz['start'], wz['end'], color='#00ff00', alpha=0.1, label='Floating Wave' if wz == wave_zones[0] else "")
        ax.text(wz['start'], ax.get_ylim()[0], ' WAVE ', color='green', fontsize=7, rotation=90)

    # Mark Questions [NEW]
    for qa in question_analysis:
        q = qa['question']
        ax.axvline(x=q['time'], color='gold', linestyle='--', alpha=0.5)
        # ax.text(q['time'], ax.get_ylim()[1], f" Q: {q['marker']}", color='#aa8800', fontsize=7, rotation=90)

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
    generate_html_report(df, incidents, calib_time, session_path, wave_zones, question_analysis)

if __name__ == "__main__":
    selected_dir = select_session()
    if selected_dir:
        print(f"Analyzing: {selected_dir}")
        df = load_data(selected_dir)
        if df is not None: plot_session(df, selected_dir)
