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
    significant = ["BLOWDOWN", "LONG FALL", "LONG RISE", "ROCKET READ", "TRANSIENT/BREATH"]
    incidents = [i for i in incidents if i['pattern'] in significant]
        
    return incidents

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

        # [NEW] Recovery Analysis (Resilience Index & Breath Detection)
        # Only applicable for DOWN movements (drops)
        recovery_search_end = min(len(df)-1, trough_idx + 100) # search ~10s ahead
        recovery_window = df.iloc[trough_idx:recovery_search_end]
        
        # T50 Recovery: Time to reach midway between trough and peak
        midpoint = trough_ta + (abs_change / 2.0)
        recovery_t50_idx = recovery_window[recovery_window['TA_Smooth'] >= midpoint].index.min()
        resilience_t50 = None
        if pd.notna(recovery_t50_idx):
            resilience_t50 = df.loc[recovery_t50_idx, 'Seconds'] - trough_time
            
        # Breath/Transient Detection: Recovers to 85% of peak
        # User defined: 3-5 seconds from peak -> trough -> peak again
        breath_threshold = trough_ta + (abs_change * 0.85)
        # Search up to 8s from peak to find the recovery peak
        recovery_window_end = min(len(df)-1, peak_idx + 80) 
        recovery_df = df.iloc[trough_idx:recovery_window_end]
        
        recovery_point_idx = recovery_df[recovery_df['TA_Smooth'] >= breath_threshold].index.min()
        is_transient = False
        if pd.notna(recovery_point_idx):
            cycle_duration = df.loc[recovery_point_idx, 'Seconds'] - peak_time
            if 2.5 <= cycle_duration <= 6.0: # Match 3-5s with slight buffer
                is_transient = True
    else: # direction == "UP"
        # Initiative defaults
        resilience_t50 = None
        is_transient = False
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

    if direction == "DOWN" and is_transient:
        new_pattern = "TRANSIENT/BREATH"

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
        'val_end': val_end,
        'resilience_t50': resilience_t50 if direction == "DOWN" else None,
        'is_transient': is_transient if direction == "DOWN" else False
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
    
    ax.set_title(f"Detail: {inc['pattern']} ({inc['abs_ta_drop']:.3f} TA)")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("TA")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_session_synopsis(duration, net_ta, incidents, question_analysis):
    """
    Generates a narrative session overview in the voice of an expert physiological data reviewer.
    """
    # 1. Net Tonic Shift (Baseline Trend)
    # net_ta is passed as total_ta_count (start_ta - end_ta)
    net_ta_val = float(net_ta)
    if net_ta_val > 0.5:
        tone = "The session exhibited a pronounced tonic decline in conductance, reflecting profound emotional resolution and the successful downregulation of the autonomic nervous system."
    elif net_ta_val < -0.5:
        tone = "The session was characterized by a distinct tonic elevation, indicative of sustained autonomic arousal and potential cognitive-emotional dissonance that remained partially unresolved."
    else:
        tone = "Tonic stability prevailed throughout the session, suggesting a state of physiological equilibrium with only minor autonomic perturbations."
        
    # 2. Phasic Reactivity (Incident Activity)
    activity_level = ""
    if len(incidents) > 15:
        activity_level = "Phasic reactivity was markedly elevated, signifying a highly volatile session with frequent episodes of acute sympathetic activation."
    elif len(incidents) < 5:
        activity_level = "The session demonstrated notably low phasic reactivity, suggesting a state of physiological stillness or, potentially, a cognitive-emotional defensive posture."
    else:
        activity_level = "Phasic reactivity was moderate and controlled, consistent with a standard processing session characterized by targeted emotional engagement."
        
    intro_paragraph = f"<p><strong>General Synopsis:</strong> In this {int(duration/60)} minute session, the participant's physiological profile revealed a nuanced interplay of tension and release. {tone} {activity_level}</p>"

    # 4. Key Observations
    observations = []
    
    significant_incidents = [i for i in incidents if 'FALL' in i['pattern'] or 'BLOWDOWN' in i['pattern']]
    if significant_incidents:
        max_drop_inc = max(significant_incidents, key=lambda x: x['abs_ta_drop'])
        if max_drop_inc['abs_ta_drop'] > 0.05:
            observations.append(f"<li><strong>Significant Phasic Release:</strong> A profound drop of {max_drop_inc['abs_ta_drop']:.3f} TA occurred at {int(max_drop_inc['start_time']//60)}:{int(max_drop_inc['start_time']%60):02}, characterized as a {max_drop_inc['pattern']}. This represents a pivotal zenith of sympathetic discharge.</li>")
        
    # Highest Cognitive-Emotional Load (Question Reactivity)
    if question_analysis:
        # Most reactive question (Max net_proc_ta magnitude)
        most_reactive = max(question_analysis, key=lambda x: abs(x['stats']['net_proc_ta']))
        q_text = most_reactive['question']['text']
        q_change = most_reactive['stats']['net_proc_ta'] # (End TA - Start TA)
        
        if q_change < -0.05:
            observations.append(f"<li><strong>Maximum Processing Yield:</strong> The stimulus '{q_text[:60]}...' elicited the most significant net resolution, resulting in a tonic decline of {abs(q_change):.3f} TA.</li>")
        elif q_change > 0.05:
            observations.append(f"<li><strong>Cognitive-Emotional Peak:</strong> The question '{q_text[:60]}...' induced the highest level of autonomic arousal, with a net elevation of {q_change:.3f} TA, highlighting a primary area of internal conflict.</li>")
        
    obs_list = "<ul>" + "".join(observations) + "</ul>" if observations else "<p>No significant phasic deviations were observed outside of baseline variance.</p>"
    
    return f"{intro_paragraph}<h4>Key Observations</h4>{obs_list}"

def generate_html_report(df, incidents, calib_time, session_path, question_analysis=None, img_b64=None):
    """Generates a comprehensive HTML report using an external template."""
    
    # User Request: Ignore PRE-calibration data for the report incidents
    if calib_time is not None:
        incidents = [inc for inc in incidents if inc['start_time'] >= calib_time]

    # Track top 5 events for summary
    significant_drops = sorted([inc for inc in incidents if 'FALL' in inc['pattern'] or 'BLOWDOWN' in inc['pattern'] or 'TRANSIENT' in inc['pattern']], 
                              key=lambda x: x['abs_ta_drop'], reverse=True)[:5]
    significant_rises = sorted([inc for inc in incidents if 'RISE' in inc['pattern'] or 'ROCKET' in inc['pattern']], 
                              key=lambda x: x['abs_ta_drop'], reverse=True)[:5]

    # 2. Metrics & Highlights
    total_duration = df['Seconds'].max()
    # Safely handle empty data
    if len(df) > 0:
        total_ta_count = df['TA'].iloc[0] - df['TA'].iloc[-1] 
    else:
        total_ta_count = 0
    major_incidents_count = len(incidents)

    # 2.5 Associate patterns with questions [NEW]
    if question_analysis:
        for qa in question_analysis:
            q_start = qa['question']['time']
            q_end = q_start + qa['stats']['duration_proc']
            # Find all incidents that started within this question's window
            qa['incidents'] = [inc for inc in incidents if q_start <= inc['start_time'] < q_end]
            
            # Cognitive Latency: Time to first incident after question start
            subsequent_incidents = [inc for inc in incidents if inc['start_time'] >= q_start]
            qa['latency'] = (subsequent_incidents[0]['start_time'] - q_start) if subsequent_incidents else None
            

    def format_highlight(inc, idx):
        mins, secs = int(inc['start_time']) // 60, int(inc['start_time']) % 60
        detail_id = f"detail_{idx}"
        return f"""
        <div class="highlight-item" onclick="var el=document.getElementById('{detail_id}'); el.style.display='block'; el.scrollIntoView();">
            <strong>{inc['pattern']}</strong> at {mins:02}:{secs:02}<br>
            <span style="font-size:1.1em; color:#d32f2f; font-weight:bold;">{inc['abs_ta_drop']:.3f} TA</span> <span style="font-size:0.8em; color:gray;">({inc['unit_drop']:.1f}U)</span>
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
        
        resilience_val = inc['resilience_t50'] if inc['resilience_t50'] is not None else 0
        resilience_str = f"{inc['resilience_t50']:.1f}s" if inc['resilience_t50'] is not None else "-"
        
        row = f"""
        <tr class="incident-row" data-pattern="{inc['pattern']}">
            <td data-val="{inc['start_time']}">{timestamp} ({phase})</td>
            <td>{q_context}</td>
            <td data-val="{inc['pattern']}">{inc['pattern']}</td>
            <td style="background:#fff9e6; font-weight:bold;" data-val="{inc['abs_ta_drop']}">{inc['abs_ta_drop']:.3f} TA</td>
            <td style="color:#666;" data-val="{inc['unit_drop']}">{inc['unit_drop']:.2f}U</td>
            <td data-val="{inc['duration']}">{inc['duration']:.2f}s</td>
            <td data-val="{resilience_val}">{resilience_str}</td>
            <td data-val="{abs(inc['abs_ta_drop']/inc['duration'])}">{abs(inc['abs_ta_drop']/inc['duration']):.3f} TA/s</td>
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
            
            latency_str = f"{qa['latency']:.2f}s" if qa['latency'] is not None else "N/A"
            latency_color = "red" if qa['latency'] is not None and qa['latency'] > 1.5 else "teal"

            q_rows += f'''
            <div id="q_detail_{idx}" class="highlight-box" style="margin-bottom:20px; border-left: 5px solid gold;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="flex:1;">
                        <span style="color:gray; font-size:0.8em;">{timestamp} ({q_data['marker']})</span>
                        <h3 style="margin:5px 0;">{q_data['text']}</h3>
                        <div class="summary-grid" style="grid-template-columns: repeat(6, 1fr); gap:10px; margin-top:10px;">
                            <div class="metric-card" style="padding:10px; border-left-color: gold;">
                                <div style="font-size:0.8em; color:gray;">15s Response</div>
                                <div style="font-size:1.0em; font-weight:bold;">{st['drop_15']:.3f} / +{st['rise_15']:.3f}</div>
                            </div>
                            <div class="metric-card" style="padding:10px; border-left-color: #d32f2f;">
                                <div style="font-size:0.8em; color:gray;">Cognitive Latency</div>
                                <div style="font-size:1.0em; font-weight:bold; color: {latency_color};">{latency_str}</div>
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
    explanation_note += "<h3>Advanced Psycho-Physiological Metrics</h3>"
    explanation_note += "<p><strong>Resilience (T50):</strong> The time in seconds to regain 50% of lost conductance after a peak. Lower T50 indicates faster emotional recovery.</p>"
    explanation_note += "<p><strong>Cognitive Latency:</strong> The delay between question delivery and the first physiological response. Fast reactions (<1s) are often subconscious, while slow reactions (>2s) may indicate cognitive filtering or repression.</p>"
    explanation_note += "<p><strong>Momentum (Velocity) Map:</strong> The color ribbon at the bottom of the main chart shows the Rate of Change (TA/s). <strong>Red</strong> indicates high friction or rapid physiological movement, even if the total shift is small. <strong>Blue</strong> indicates stable processing.</p>"
    explanation_note += "<p><strong>TRANSIENT/BREATH:</strong> Automatically detected breath artifacts. These are 3-5s drops that recover to >85% of their original baseline almost immediately.</p>"
    explanation_note += "</div>"

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
    synopsis_html = generate_session_synopsis(total_duration, total_ta_count, incidents, question_analysis)
    html_content = html_content.replace("{{ SESSION_OVERVIEW }}", synopsis_html) 
    
    html_content = html_content.replace("{{ HIGHLIGHTS_SECTION }}", highlights_html)
    html_content = html_content.replace("{{ QUESTION_SECTION }}", question_html)
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
    incidents = analyze_incidents(df)
    
    # Filter by calibration time [REQ]
    if calib_time is not None:
        incidents = [inc for inc in incidents if inc['start_time'] >= calib_time]

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
     
    # [NEW] Momentum Heatmap Ribbon (Rate of Change)
    # 0 = Blue (Stable), 0.25 TA/s = Red (High Friction)
    df_m = df.copy()
    diff_s = df_m['Seconds'].diff().fillna(0.1)
    diff_s[diff_s == 0] = 0.1
    df_m['Velocity_Raw'] = df_m['TA'].diff().abs().fillna(0) / diff_s
    df_m['Velocity_Heat'] = df_m['Velocity_Raw'].rolling(window=30).mean().fillna(0)
    norm = plt.Normalize(0, 0.25) 
    cmap = plt.get_cmap('coolwarm')
    
    # Draw ribbon at the bottom 3% of the plot
    step = max(2, len(df_m) // 250) 
    for i in range(0, len(df_m)-step, step):
        v = df_m['Velocity_Heat'].iloc[i]
        ax.axvspan(df_m['Seconds'].iloc[i], df_m['Seconds'].iloc[min(len(df_m)-1, i+step)], 
                   ymin=0, ymax=0.03, color=cmap(norm(v)), alpha=0.9)
    
    ax.text(df_m['Seconds'].min(), 0, " MOMENTUM (VELOCITY) ", color='white', fontsize=8, 
            fontweight='bold', transform=ax.get_xaxis_transform(), va='bottom', backgroundcolor='#444')

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
    
    # Get base64 for HTML report
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    generate_html_report(df, incidents, calib_time, session_path, question_analysis, img_b64)

if __name__ == "__main__":
    selected_dir = select_session()
    if selected_dir:
        print(f"Analyzing: {selected_dir}")
        df = load_data(selected_dir)
        if df is not None: plot_session(df, selected_dir)
