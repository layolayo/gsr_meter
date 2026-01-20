
import asyncio
import threading
import collections
import csv
import time
import math
import queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons, RadioButtons, Button, Slider, TextBox
from datetime import datetime, timedelta
import os
import json
import tkinter as tk
from tkinter import ttk
import hid
from modules.audio_handler import AudioHandler
from modules.ant_driver import AntHrvSensor
from modules.hrv_state_analyzer import HRVStateAnalyzer
# Pattern Rec
from modules.gsr_patterns import GSRPatterns 


# --- CONFIGURATION ---
FS = 250 # Hertz of headset datastream
# Trend Smoothing Window (User variable)

TREND_WINDOW_SEC = 2.0 # Default 1s smoothing
SMOOTHING_SAMPLES = int(FS * TREND_WINDOW_SEC)

BUFFER_SIZE = int(FS * 10) 
MAX_DISPLAY_SEC = 10 # Show 20 seconds of data on screen


WARMUP_SAMPLES = FS * 3
SCORE_SMOOTHING_WINDOW = int(FS * 2.0) 

# File Naming
CONFIG_FILE = "v37_config.json"

# --- GSR SETTINGS ---
VENDOR_ID = 0x1fc9
PRODUCT_ID = 0x0003
V_SOURCE = 6.371
R_REF = 83.0

# --- GLOBAL VARIABLES ---
latest_gsr_ta = 0.0
GSR_CENTER_VAL = 3.0
BASE_SENSITIVITY = 0.3 
CALIB_TARGET_RATIO = 1.0 / 2.0 
booster_level = 0      # 0=OFF, 1=LO, 2=MED, 3=HI
active_boost_level = 0 # The level currently applied to graph zoom
CALIB_PIVOT_TA = 2.0
active_event_label = ""   # The TA value where Sensitivity scaling is neutral (1.0x)
last_stable_window = 5.0  # [NEW] Track last valid window for motion freeze
gsr_capture_queue = collections.deque(maxlen=100)
# [NEW] Manual Blit Globals
graph_bg = None 
bg_scores = None
bg_count  = None
bg_detail = None
bg_status = None
bg_sens   = None
bg_info = None # [FIX] Background for System Line
first_run_center = False # [FIX] Init Auto-Center flag

# --- HRM GLOBALS ---
latest_hr = 0
latest_hrv = 0.0
hrm_status = "Init"
hrm_sensor = None
hrv_analyzer = None # [NEW]
latest_hrm_state = None # [NEW]
txt_hr_val = None
txt_hrv_val = None
gsr_patterns = None # [NEW] Engine
current_pattern = "IDLE"


# [NEW] Global Motion Detector (Velocity Filter)
motion_lock_expiry = 0.0
motion_prev_ta = 0.0
motion_prev_time = 0.0
motion_lock_active = False # [NEW] State tracker for logging
session_start_time = time.time() # [NEW] Track app start for Warmup

def get_effective_window():
    # Use active_boost_level (applied on Reset) not target booster_level
    if active_boost_level == 0: 
        return min(1.0, BASE_SENSITIVITY) 
    
    # Logic from v20
    mult = [1.0, 0.6, 1.0, 1.4][active_boost_level]
    safe_ta = max(1.0, GSR_CENTER_VAL)
    global last_stable_window, motion_lock_expiry    

    # Motion Safety: If Locked, FREEZE sensitivity at last known good value.

    if time.time() < motion_lock_expiry:
         return last_stable_window

    try:
        # Ratio = TA / CALIB_PIVOT. (e.g. 2.75 / 2.75 = 1.0)
        # We want Higher TA -> Smaller Window (Higher Sens).
        # So we raise to NEGATIVE mult. (Ratio ^ -mult).
        # If Ratio = 1.0 (Reset), Factor = 1.0 -> No Change.
        pivot = max(0.1, CALIB_PIVOT_TA) # Safety
        val = BASE_SENSITIVITY * math.pow((safe_ta / pivot), -mult)
        
        # Enforce Global Limit: Window <= 1.0 (Sens >= 1.0)
        # Even if Auto-Boost zooms out (because TA < Pivot), don't go below Sens 1.0

        val = min(1.0, val)
        
        last_stable_window = val # Update stable value
        return val
    except:
        return BASE_SENSITIVITY

# --- STATE ---
# 60Hz * 10s = 600 samples (60Hz Plotting) -> Increased to 800 to ensure full coverage
HISTORY_LEN = 800
bands_history = {k: collections.deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN) for k in ['GSR']}

log_messages = collections.deque(["System Ready"], maxlen=10)
app_running = True
counting_active = False # TA Counter State
ta_accum = 0.0          # TA Counter Total
is_recording = False
last_motion_log = 0.0
is_recording = False
samples_since_contact = 0
device_battery_level = "--"
ui_update_queue = queue.Queue()
command_queue = queue.Queue()

# Biofeedback
current_calm_score = 0
calm_history = collections.deque(maxlen=SCORE_SMOOTHING_WINDOW)

# Logic Settings
event_detected = False
total_samples_recorded = 0

# Control Defaults
global_percent = 20

# CSV Handles
f_gsr = None
writer_gsr = None
recording_start_time = None

# --- AUDIO RECORDING STATE ---
audio_filename = None

# ==========================================
#           GSR READER 
# ==========================================
class GSRReader(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        self.current_ta = 0.0
        self.connected = False
        self.connected = False
        # Internal Buffer for High-Res Processing
        self.eeg_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.samples_to_process = 0.0 
        self.last_flush_time = 0 


    def run(self):
        try:
            global latest_gsr_ta, current_calm_score
            h = hid.device()
            h.open(VENDOR_ID, PRODUCT_ID)
            h.set_nonblocking(1) 
            self.connected = True
            print("[GSR] Connected (High Speed Mode)")

            while self.running:
                try:
                    data = None
                    while True:
                        try:
                            d = h.read(64)
                            if d: data = d
                            else: break
                        except: break
                    
                    if not data:
                        time.sleep(0.005) 
                        continue
                    
                    if len(data) >= 4 and data[0] == 0x01:
                        self.connected = True
                        raw_val = (data[2] << 8) | data[3]
                        voltage = raw_val / 10000.0
                        
                        if voltage >= (V_SOURCE - 0.005):
                            ohms = 999999.9
                        else:
                            try:
                                ohms = (voltage * R_REF) / (V_SOURCE - voltage)
                            except:
                                ohms = 999999.9
                        try:
                            ta = (ohms * 1000 / (ohms * 1000 + 21250)) * 5.559 + 0.941
                        except:
                            ta = 0.0
                        self.current_ta = ta
                        
                    global latest_gsr_ta
                    latest_gsr_ta = self.current_ta
                    
                    # Log Here to capture every sample (60Hz)

                    if is_recording:
                         try:
                             ts_now = datetime.now().strftime('%H:%M:%S.%f')
                             
                             # Log GSR
                             if writer_gsr:
                                  win = get_effective_window()
                                  global CALIB_PIVOT_TA, active_boost_level, active_event_label, motion_lock_expiry, current_pattern
                                  note = active_event_label
                                  if note: active_event_label = ""
                                  
                                  is_motion = 1 if time.time() < motion_lock_expiry else 0
                                  
                                  # Calc Elapsed
                                  elapsed = 0.0
                                  if recording_start_time:
                                       elapsed = (datetime.now() - recording_start_time).total_seconds()
                                       
                                  writer_gsr.writerow([ts_now, f"{elapsed:.3f}", f"{self.current_ta:.5f}", f"{GSR_CENTER_VAL:.3f}", f"{1.0/win:.3f}", f"{win:.3f}", is_motion, f"{CALIB_PIVOT_TA:.3f}", active_boost_level, note, current_pattern])
                             
                         except Exception as e: pass

                    # Periodic Flush to prevent IO Stalls (Every 1.0s)
                    if is_recording and (time.time() - self.last_flush_time) > 1.0:
                        try:
                            if f_gsr: f_gsr.flush()
                            if f_hrm: f_hrm.flush()
                            self.last_flush_time = time.time()
                        except: pass

                    # GRAPH HISTORY UPDATE (Master Clock = 60Hz)
                    try:
                         # 1. GSR Value
                         # Store RAW TA for dynamic scaling in main loop
                         bands_history['GSR'].append(self.current_ta)

                         
                    except: pass
                
                except Exception as loop_e:
                    print(f"[GSR] Loop Skip: {loop_e}")
                    time.sleep(0.005)
 
                time.sleep(0.0166) # ~60Hz Pacing 
 

        except Exception as e:
            print(f"[GSR] FATAL Error: {e}") 
            self.connected = False

    def stop(self):
        self.running = False

def log_msg(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    clean_msg = str(msg).strip()
    log_messages.append(f"[{timestamp}] {clean_msg}")
    
    # Update UI directly if available
    target = None
    if 'ui_refs' in globals():
         if 'system_line' in ui_refs: target = ui_refs['system_line']
         elif 'log_text' in ui_refs: target = ui_refs['log_text']
         
    if target:
        try: target.set_text(clean_msg)
        except: pass
        
# --- CONFIG PERSISTENCE ---
def save_config():
    try:
        # Save Graph Visibility from Checkbuttons

        # We need a way to access the current visibility state.
        # Since 'check' is inside the GUI setup, we might need a global ref or read from 'lines'.
        # Assuming 'lines' is global/accessible (it is in main view setup, but lines dict is populated in main).
        # We need to make sure 'lines' is accessible here. It is local to main block.
        # Wait, save_config is global. lines is defined in `if __name__ == "__main__":`.
        # We can't access `lines` directly if it's local.
        # However, `check_status` is not easily accessible unless we make it global.
        # Let's inspect line 700 area again.
        
        # ACTUALLY, we can save the Global Variables for GSR.
        vis_state = {}
        # We need to access the 'check' widget or 'lines' to know what is visible.
        # Solution: Use a global `active_graph_lines` list or similar.
        # Let's inspect line 700 area again.
        
        cfg = {
            'mic_name': audio_handler.current_mic_name if 'audio_handler' in globals() else "Default",
            'mic_gain': audio_handler.current_mic_gain if 'audio_handler' in globals() else 3.0,
            'mic_rate': audio_handler.current_mic_rate if 'audio_handler' in globals() else None, 
            'gsr_center': GSR_CENTER_VAL,
            'gsr_base': BASE_SENSITIVITY, 
            'booster_idx': booster_level  
        }
        
        # (Graph Visibility logic removed)
            
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=4)
        print(f"[Config] Saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"[Config] Save Error: {e}")

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print("[Config] No file found. Using defaults.")
        return

    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)

        if 'audio_handler' in globals():
             audio_handler.current_mic_name = cfg.get('mic_name', "Default")
             audio_handler.current_mic_gain = float(cfg.get('mic_gain', 3.0))
             audio_handler.current_mic_rate = cfg.get('mic_rate', None)
             print(f"[Config] Audio: {audio_handler.current_mic_name}, Rate: {audio_handler.current_mic_rate}, Gain: {audio_handler.current_mic_gain}")

        # Restore GSR Settings
        global GSR_CENTER_VAL, BASE_SENSITIVITY, booster_level
        GSR_CENTER_VAL = float(cfg.get('gsr_center', 3.0))
        BASE_SENSITIVITY = float(cfg.get('gsr_base', 0.3)) 
        if 'gsr_base' not in cfg: BASE_SENSITIVITY = float(cfg.get('gsr_window', 0.3))
            
        booster_level = int(cfg.get('booster_idx', 0))
        
        print(f"[Config] Loaded settings.")
    except Exception as e:
        print(f"[Config] Load Error: {e}")

if __name__ == "__main__":
    # Globals Init
    f_hrm = None
    writer_hrm = None

    # Start GSR Thread
    gsr_thread = GSRReader()
    gsr_thread.start()
    
    # Start HRM Sensor
    try:
        hrm_sensor = AntHrvSensor()
        hrm_sensor.start()
        # Initialize Analyzer
        hrv_analyzer = HRVStateAnalyzer()
        print("[HRM] Sensor Started")

    except Exception as e:
        print(f"[HRM] Start Error: {e}")  

    pass 
    
    # --- GUI ---
    fig = plt.figure(figsize=(15, 9)) 
    try: fig.canvas.manager.set_window_title("EK GSR/HRM Session Monitor")

    except: pass 
    
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05)
    
    current_view = 'main'
    desired_view = 'main'
    ax_positions = {}
    main_view_axes = []
    settings_view_axes = []
    ui_refs = {} # [FIX] Init global refs early
    
    # Audio Handler
    def update_audio_ui(key, val):
        try:
            if key == 'mic_name_text' and 'text_mic_name' in ui_refs: 
                 ui_refs['text_mic_name'].set_text(val)
            elif key == 'status_text': 
                 if 'txt_audio' in ui_refs: ui_refs['txt_audio'].set_text(val)
                 elif 'txt_audio' in globals(): txt_audio.set_text(val)
        except: pass
        
    audio_handler = AudioHandler(log_msg, update_audio_ui)
    
    # Load Config AFTER AudioHandler exists
    load_config()

    # [FIX] Probe Audio Immediately (Now that Handler + Config are ready)
    print("[Startup] Probing Audio Device...")
    audio_handler.sync_audio_stream('main') 
    
    # Calibration State
    calib_mode = False
    calib_phase = 0 # 0=Wait, 1=Drop, 2=Recover, 3=Stabilize
    calib_step = 0
    calib_start_time = 0
    calib_base_ta = 0.0
    calib_min_ta = 0.0
    calib_vals = []
    
    def reg_ax(rect, group_list):
        ax = plt.axes(rect)
        ax_positions[ax] = rect
        group_list.append(ax)
        return ax
    
    def create_panel_ax(rect, title):
        ax_p = reg_ax(rect, settings_view_axes)
        ax_p.set_xticks([]); ax_p.set_yticks([]); ax_p.set_facecolor('#e0e0e0')
        ax_p.text(0.05, 0.95, title, transform=ax_p.transAxes, ha='left', va='top', fontsize=9, fontweight='bold', color='#333')
        return ax_p

    # === ROLLING LINE GRAPH (Main Area) ===
    ax_graph = reg_ax([0.05, 0.20, 0.78, 0.70], main_view_axes)
    ax_graph.set_xlim(0, HISTORY_LEN)
    ax_graph.set_ylim(-5, 105)
    ui_refs['ax_graph'] = ax_graph 
    
    ax_graph.set_title("GSR Monitor ~10s", fontsize=14, fontweight='bold')
    ax_graph.set_xticks([0,80,160,240,320,400,480,560,640,720,800])
    ax_graph.set_xticklabels(["10","9","8","7","6","5","4","3","2","1","0"])
    
    # [FIX] Ensure Spines are ON TOP of the data line (Z=100)
    for spine in ax_graph.spines.values():
        spine.set_zorder(100)

    # Create lines
    lines = {}
   
    line, = ax_graph.plot([], [], lw=2, color='magenta', label='GSR')
    lines['GSR'] = line
    ax_graph.axhline(y=62.5, color='#CC5500', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    txt_ta_set_line = ax_graph.text(0, 63.0, f"TA SET: {GSR_CENTER_VAL:.2f}", color='#CC5500', fontsize=8, fontweight='bold', ha='left')
    # [NEW] Manual Blit Background Capture
    def on_draw(event):
        global graph_bg, bg_scores, bg_count, bg_detail, bg_status, bg_sens, bg_info
        if fig:
            if ax_graph: graph_bg = fig.canvas.copy_from_bbox(ax_graph.bbox)
            if ax_scores: bg_scores = fig.canvas.copy_from_bbox(ax_scores.bbox)
            if ax_count_bg: bg_count = fig.canvas.copy_from_bbox(ax_count_bg.bbox)
            if ax_detail: bg_detail = fig.canvas.copy_from_bbox(ax_detail.bbox)
            if ax_status: bg_status = fig.canvas.copy_from_bbox(ax_status.bbox)
            if ax_w_val:  bg_sens   = fig.canvas.copy_from_bbox(ax_w_val.bbox)
            
            # [FIX] Capture Info Panel too
            if ax_info and current_view == 'main': bg_info = fig.canvas.copy_from_bbox(ax_info.bbox)
            
    fig.canvas.mpl_connect('draw_event', on_draw)
    
    # [FIX] Set Animated (Exclude from Background)
    if 'txt_ta_set_line' in locals(): locals()['txt_ta_set_line'].set_animated(True)
    if line: line.set_animated(True)
    
    # [FIX] Restore Overlay Axis for Global Text
    ax_overlay = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax_overlay.set_xticks([]); ax_overlay.set_yticks([])
    ax_overlay.set_zorder(100) # Ensure on top
    
    # [NEW] Calibration Overlay Text (Ax Level)
    txt_calib_overlay = ax_overlay.text(0.5, 0.5, "", ha='center', va='center', fontsize=24, fontweight='bold', color='red')
    ui_refs['txt_calib_overlay'] = txt_calib_overlay 
    txt_calib_overlay.set_animated(True) # [FIX] Animated
    
    # [NEW] Motion Overlay Text (Top Center)
    txt_motion_overlay = ax_overlay.text(0.5, 0.75, "", ha='center', va='center', fontsize=20, fontweight='bold', color='red')
    ui_refs['txt_motion_overlay'] = txt_motion_overlay
    txt_motion_overlay.set_animated(True) # [FIX] Animated
    
    # [NEW] GSR Pattern Text (Bottom Center of Graph)
    txt_pattern = ax_graph.text(0.5, 0.02, "PATTERN: IDLE", transform=ax_graph.transAxes, ha='center', va='bottom', fontsize=14, fontweight='bold', color='gray', zorder=90)
    txt_pattern.set_animated(True) # [FIX] Prevent Ghosting
    ui_refs['txt_pattern'] = txt_pattern
    
    # === GSR CONTROLS CONTAINER ===
    # A single bordered panel to contain Scale, Sens, Boost, Calibrate
    r_ctrl = [0.835, 0.57, 0.13, 0.33] # H = 0.33
    ax_ctrl_bg = reg_ax(r_ctrl, main_view_axes)
    ax_ctrl_bg.set_facecolor('#f9f9f9')
    ax_ctrl_bg.set_xticks([]); ax_ctrl_bg.set_yticks([])
    # Border
    rect_ctrl_border = plt.Rectangle((0,0), 1, 1, transform=ax_ctrl_bg.transAxes, fill=False, ec='#aaaaaa', lw=2, clip_on=False)
    ax_ctrl_bg.add_patch(rect_ctrl_border)
    
    # --- 1. Title: GSR Scale ---
    # Relative to main axes to align easily
    ax_scale_lbl = reg_ax([0.835, 0.85, 0.13, 0.04], main_view_axes)
    ax_scale_lbl.set_axis_off()
    ax_scale_lbl.text(0.5, 0.5, "GSR Scale", ha='center', fontweight='bold', fontsize=12)
    
    # --- 2. Sensitivity ---
    ax_win_lbl = reg_ax([0.835, 0.81, 0.13, 0.03], main_view_axes)
    ax_win_lbl.set_axis_off()
    ax_win_lbl.text(0.5, 0.5, "Sensitivity", ha='center', va='center', fontsize=10, fontweight='bold', color='#444')
    
    # Stepper [ - ] [ Val ] [ + ]
    y_sens = 0.77 # [MOVED UP] Was 0.50
    ax_w_down = reg_ax([0.85, y_sens, 0.03, 0.03], main_view_axes)
    ax_w_val  = reg_ax([0.88, y_sens, 0.04, 0.03], main_view_axes)
    ax_w_up   = reg_ax([0.92, y_sens, 0.03, 0.03], main_view_axes)
    
    # [FIX] Enforce Z-Order > 50 (Proxy Layer)
    ax_w_down.set_zorder(100); ax_w_val.set_zorder(100); ax_w_up.set_zorder(100)
    
    ax_w_val.set_axis_off()
    
    def get_display_sens():
        w = get_effective_window()
        if w <= 0.001: return 99.9
        return 1.0 / w

    # Initial display
    txt_win_val = ax_w_val.text(0.5, 0.5, f"{get_display_sens():.2f}", ha='center', va='center', fontsize=10, fontweight='bold')
    txt_win_val.set_animated(True)
    
    btn_win_down = Button(ax_w_down, "-", color='lightgray', hovercolor='gray')
    btn_win_up   = Button(ax_w_up, "+", color='lightgray', hovercolor='gray')
    
    # --- 3. Auto-Boost ---
    ax_boost_lbl = reg_ax([0.835, 0.72, 0.13, 0.02], main_view_axes)
    ax_boost_lbl.set_axis_off()
    ax_boost_lbl.text(0.5, 0.5, "Auto-Boost", ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    # Buttons [OFF] [L] [M] [H]
    y_b = 0.68 # [MOVED UP] Was 0.41
    ax_b_off = reg_ax([0.845, y_b, 0.030, 0.025], main_view_axes)
    ax_b_lo  = reg_ax([0.880, y_b, 0.020, 0.025], main_view_axes)
    ax_b_med = reg_ax([0.905, y_b, 0.020, 0.025], main_view_axes)
    ax_b_hi  = reg_ax([0.930, y_b, 0.020, 0.025], main_view_axes)
    
    # [REQ] 0 -> OFF
    btn_b_off = Button(ax_b_off, "OFF", color='lightgray', hovercolor='gray')
    btn_b_off.label.set_fontsize(7)
    
    btn_b_lo  = Button(ax_b_lo,  "L", color='lightgray', hovercolor='gray')
    btn_b_med = Button(ax_b_med, "M", color='lightgray', hovercolor='gray')
    btn_b_hi  = Button(ax_b_hi,  "H", color='lightgray', hovercolor='gray')
    
    boost_btns = [btn_b_off, btn_b_lo, btn_b_med, btn_b_hi]
    
    def update_boost_ui():
        for i, btn in enumerate(boost_btns):
            if i == booster_level:
                # Active
                btn.color = 'lime'
                btn.hovercolor = 'lime'
                try: btn.label.set_color('black')
                except: pass
            else:
                # Idle
                btn.color = 'lightgray'
                btn.hovercolor = 'gray'
                try: btn.label.set_color('black')
                except: pass

        
    update_boost_ui() # Init
    
    def set_boost(lvl):
        global booster_level, active_boost_level
        booster_level = lvl
        # active_boost_level = lvl # [REQ] Removed immediate sync. Waits for Reset.
        
        update_boost_ui()
        # Update text immediately (Shows potential, or current? Current Window usually)
        # But if active != booster, display might differ. 
        # txt_w_val shows get_display_sens().
        txt_win_val.set_text(f"{get_display_sens():.2f}")
        
        # [FIX] Do NOT Reset immediately. User wants Auto-Boost to only trigger on OOB.
        # if latest_gsr_ta > 0.01:
        #      update_gsr_center(float(f"{latest_gsr_ta:.2f}"))             
        #plt.draw()

    btn_b_off.on_clicked(lambda e: set_boost(0))
    btn_b_lo.on_clicked(lambda e: set_boost(1))
    btn_b_med.on_clicked(lambda e: set_boost(2))
    btn_b_hi.on_clicked(lambda e: set_boost(3))
    
    # --- 4. Calibrate ---
    ax_calib = reg_ax([0.85, 0.60, 0.10, 0.04], main_view_axes)
    ax_calib.set_zorder(100) # [FIX] Ensure Clickable
    
    btn_calib = Button(ax_calib, "Calibrate", color='lightblue', hovercolor='aquamarine')
    
    # Saved Boost Level for Restore
    global saved_boost_level
    saved_boost_level = 0
    
    def start_calibration(e):
        global calib_mode, calib_step, calib_phase, calib_start_time, calib_base_ta, calib_min_ta, calib_vals, calib_step_start_time
        global booster_level, saved_boost_level, CALIB_PIVOT_TA, active_boost_level
        
        # Save current level
        saved_boost_level = booster_level
        # Reset Pivot to default (Optional, but clean)
        CALIB_PIVOT_TA = 2.0
        
        # Force Manual Mode (OFF)
        set_boost(0)
        active_boost_level = 0 # [REQ] Force Graph to Manual Mode immediately 
        
        # [REQ] Auto-Set Sensitivity to 2.0 (Window 0.5)
        global BASE_SENSITIVITY
        BASE_SENSITIVITY = 0.5
        txt_win_val.set_text(f"{get_display_sens():.2f}")
        #plt.draw() 
        
        calib_mode = True
        calib_phase = 0
        calib_step = 1
        calib_start_time = time.time()
        calib_vals = [] # [NEW] Median Buffer
        calib_base_ta = latest_gsr_ta
        calib_min_ta = latest_gsr_ta
        calib_vals = [] # Store drops
        
        log_msg(f"Calibration Started. Saving Boost Lvl: {saved_boost_level}")
        global active_event_label
        active_event_label = "CALIB_START"
        update_gsr_center(latest_gsr_ta)
        
    btn_calib.on_clicked(start_calibration)
    
    # --- 5. BIO-GRID (Start at Bottom) ---
    r_bio = [0.835, 0.38, 0.13, 0.17] 
    ax_grid = reg_ax(r_bio, main_view_axes)
    ax_grid.set_facecolor('#dddddd')
    ax_grid.set_xlim(-6, 6) # Z-Score Range (Expanded)
    ax_grid.set_ylim(-6, 6)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    
    # Crosshair
    ax_grid.axhline(0, color='gray', lw=1, alpha=0.5)
    ax_grid.axvline(0, color='gray', lw=1, alpha=0.5)
    
    # Labels (Small)
    # [FIX] Swapped FLOW (Q2, Top-Right) and STRESS (Q1, Top-Left)
    ax_grid.text(0.15, 0.90, "STRESS", transform=ax_grid.transAxes, fontsize=6, color='red', alpha=0.5)
    ax_grid.text(0.15, 0.05, "WDRAW", transform=ax_grid.transAxes, fontsize=6, color='gray', alpha=0.5)
    ax_grid.text(0.65, 0.90, "FLOW", transform=ax_grid.transAxes, fontsize=6, color='green', alpha=0.5)
    ax_grid.text(0.65, 0.05, "RECOV", transform=ax_grid.transAxes, fontsize=6, color='blue', alpha=0.5)
    
    # Moving Dot
    grid_dot, = ax_grid.plot([], [], 'o', color='purple', markersize=8, markeredgecolor='black')
    
    # Text Below
    ax_grid_txt = reg_ax([0.835, 0.29, 0.13, 0.08], main_view_axes)
    ax_grid_txt.set_axis_off()
    txt_grid_state = ax_grid_txt.text(0.5, 0.70, "Waiting...", ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    txt_grid_trend = ax_grid_txt.text(0.5, 0.30, "--", ha='center', va='center', fontsize=9, color='#444')
    
    def update_grid(z_hr, z_hrv, state, trend):
        # [FIX] Removed try/except to expose errors.
        # [FIX] Cast inputs to float() because Analyzer returns strings for CSV consistency.
        try:
            val_h = float(z_hr)
            val_v = float(z_hrv)
        except:
            val_h = 0.0
            val_v = 0.0
            
        z_x = max(-6.0, min(6.0, val_v)) # Note: X is HRV (Rmssd)
        z_y = max(-6.0, min(6.0, val_h)) # Note: Y is HR (BPM)
        
        grid_dot.set_data([z_x], [z_y])
        
        # Color based on quadrant?
        # Q1 (Stress), Q2 (Flow)...
        if val_h >= 0 and val_v >= 0: c = 'green'
        elif val_h >= 0 and val_v < 0: c = 'red' 
        elif val_h < 0 and val_v >= 0: c = 'blue'
        else: c = 'gray'
        
        # [NEW] Balanced State Override
        if state == "BALANCED":
            c = 'black' 
        
        grid_dot.set_color(c)
        
        txt_grid_state.set_text(state.upper())
        txt_grid_state.set_color(c)
        txt_grid_trend.set_text(trend)


    
    def change_sensitivity(direction):
        global BASE_SENSITIVITY
        # [REQ] Increase/Decrease by % of current TA SET (GSR_CENTER_VAL)
        # "delta" was 0.05. Now we interpret direction as +1 or -1 sign.
        # Step size = 0.5% of Center (Factor of 10 less than 5%).
        step = max(0.005, GSR_CENTER_VAL * 0.005)
        
        # If direction > 0 (+ button), likely means INCREASE SENSITIVITY (make graph more reactive = SMALLER WINDOW)
        # But previous code said: "Increase Sens = Subtract Window".
        # So: New Window = Base - (Direction * Step)
        
        # [FIX] Sensitivity >= 1.0 means Window <= 1.0
        # Mult is an exponent, so at Pivot (Ratio=1), it has no effect.
        # We just clamp Base Window to 1.0.
        
        delta = step * direction
        new_win = BASE_SENSITIVITY - delta
        
        BASE_SENSITIVITY = max(0.05, min(1.0, rounded_step(new_win)))
        txt_win_val.set_text(f"{get_display_sens():.2f}")
        #plt.draw()
        
    def rounded_step(val): return round(val * 1000) / 1000.0 # Round to nearest 0.001

    # [FIX] Inverted Buttons: Up(+) adds Sensitivity (Reduces Window)
    btn_win_down.on_clicked(lambda e: change_sensitivity(-1)) # Reduce Sens = Add to Window
    btn_win_up.on_clicked(lambda e: change_sensitivity(1))    # Increase Sens = Subtract Window

    # [FIX] Moved TA SET to central box
    val_txt = None # Placeholder, defined below

    # [FIX] Added last_calib_ratio for debug
    last_calib_ratio = 0.0

    def update_gsr_center(val, force_pivot=False, reason="System"):
        global GSR_CENTER_VAL, ta_accum, calib_mode # [FIX] Added ta_accum, calib_mode
        global active_boost_level, booster_level    # [FIX] Added boost globals
        global motion_lock_expiry # [NEW]
        global CALIB_PIVOT_TA
        
        # Log the Update
        if reason != "System":
             log_msg(f"{reason}: Center to {val:.2f}")

        # [NEW] TA Counter Logic (Count Drops)
        if counting_active:
             # [REQ] Global Motion Block
             if time.time() > motion_lock_expiry:
                 diff = GSR_CENTER_VAL - val
                 if diff > 0 and not calib_mode: 
                     ta_accum += diff

        if not calib_mode:
            # Update Pivot ONLY if Forced (Calibration Complete)
            if force_pivot:
                 CALIB_PIVOT_TA = max(0.1, val)
                 log_msg(f"Pivot FORCE Set to {CALIB_PIVOT_TA:.2f}")
            else:
                 # Standard Reset (User or Auto)
                 # Only update Pivot if Reference Reset (Space/Button)
                 if reason.startswith("User"):
                      CALIB_PIVOT_TA = max(0.1, val)

            active_boost_level = booster_level

        GSR_CENTER_VAL = val
        if val_txt: val_txt.set_text(f"TA SET: {val:.2f}")

    # Keyboard Control
    def on_key(event):
        global GSR_CENTER_VAL, active_boost_level, booster_level
        eff_win = get_effective_window() # [FIX] Use effective window
        # Inverted Logic:
        # Intuition: "Up" arrow = Increase Value.
        if event.key == 'up':
            new_val = min(8.0, GSR_CENTER_VAL - (eff_win * 0.1))
            update_gsr_center(new_val)
        elif event.key == 'down':
            new_val = max(0.0, GSR_CENTER_VAL + (eff_win * 0.1))
            update_gsr_center(new_val)
        elif event.key == 'right':
            change_sensitivity(1) # [FIX] +Sens (Decrease Window)
        elif event.key == 'left':
            change_sensitivity(-1) # [FIX] -Sens (Increase Window)
        elif event.key == ' ':
            # [REQ] Spacebar auto-sets TA Center to current value AND syncs Auto-Boost Zoom
            if latest_gsr_ta > 0.1:
                global active_event_label
                if gsr_patterns:
                    pass # gsr_patterns.reset() # [FIX] Disabled per user request
                
                # [REQ] Log the reset event
                active_event_label = f"USER_TA_RESET to {latest_gsr_ta:.3f}"
                # log_msg moved to update_gsr_center
                active_boost_level = booster_level
                update_gsr_center(latest_gsr_ta, reason="User Reset")

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # === TA & SCORES PANELS (Clean Re-implementation) ===
    # Moved UP to avoid overlapping System Info (Y=0.04)
    # Graph Bottom is now 0.20, so we have 0.04 to 0.20 to work with (Height 0.16)
    
    # 1. TA Counter Panel (Right Side of Left Block)
    r_count = [0.20, 0.05, 0.20, 0.12]
    ax_count_bg = reg_ax(r_count, main_view_axes)
    ax_count_bg.set_xticks([]); ax_count_bg.set_yticks([])
    
    # Background Patch (Dynamic Color)
    bg_count_rect = plt.Rectangle((0,0), 1, 1, transform=ax_count_bg.transAxes, color='#ffcccc', ec='black', lw=2, clip_on=False)
    bg_count_rect.set_zorder(0) # Patch at bottom of Axes
    ax_count_bg.add_patch(bg_count_rect)
    ui_refs['count_bg_rect'] = bg_count_rect
    
    # Text
    txt_count_val = ax_count_bg.text(0.5, 0.70, "TA Counter: 0.00", ha='center', va='center', fontsize=16, fontweight='bold', color='#550000')
    txt_count_val.set_zorder(10)
    txt_count_val.set_animated(True)
    
    # Buttons (Axes inside the panel area relative to Figure)
    ax_btn_start = reg_ax([0.23, 0.06, 0.07, 0.04], main_view_axes)
    ax_btn_reset = reg_ax([0.31, 0.06, 0.07, 0.04], main_view_axes)
    
    # Ensure Z-Order (Axes Level)
    ax_count_bg.set_zorder(1)
    ax_btn_start.set_zorder(100)
    ax_btn_reset.set_zorder(100)
    
    # Start/Reset Buttons
    
    ui_refs['btn_count'] = Button(ax_btn_start, 'Start', color='#ccffcc', hovercolor='lime') 
    ui_refs['btn_count'].label.set_color('black')
    
        
    ui_refs['btn_count'] = Button(ax_btn_start, 'Start', color='#ccffcc', hovercolor='lime') 
    ui_refs['btn_count'].label.set_color('black')
    # [FIX] Set Animated (Draw Manually)
    ui_refs['btn_count'].ax.patch.set_animated(True)
    ui_refs['btn_count'].label.set_animated(True)
    
        
    ui_refs['btn_reset'] = Button(ax_btn_reset, 'Reset', color='#ffcccc', hovercolor='red')
    ui_refs['btn_reset'].label.set_color('black')
    # [FIX] Set Animated
    ui_refs['btn_reset'].ax.patch.set_animated(True)
    ui_refs['btn_reset'].label.set_animated(True)
    
    # Logic
    def toggle_count(e):
        global counting_active
        counting_active = not counting_active
        
        # Colors
        c_bg = '#ccffcc' if counting_active else '#ffcccc'
        c_fg = '#005500' if counting_active else '#550000'
        bg_count_rect.set_facecolor(c_bg)
        txt_count_val.set_color(c_fg)
        
        # Button State
        b = ui_refs['btn_count']
        b.label.set_text("Stop" if counting_active else "Start")
        
        # Active(Stop) -> Salmon (Red). Idle(Start) -> LightGreen
        # Direct Button Update
        if counting_active:
             b.color = 'salmon'
             b.hovercolor = 'red'
        else:
             b.color = '#ccffcc'
             b.hovercolor = 'lime'
        
       
        # Disable Reset if counting
        br = ui_refs['btn_reset']
        if counting_active:
            br.color = '#eeeeee'
            br.label.set_color('gray')
            br.hovercolor = '#eeeeee'
        else:
            br.color = '#ffcccc'
            br.label.set_color('black')
            br.hovercolor = 'red'

        # fig.canvas.draw_idle()

    def reset_count(e):
        if counting_active: return
        global ta_accum
        ta_accum = 0.0
        txt_count_val.set_text(f"TA Counter: {ta_accum:.2f}")
        # fig.canvas.draw_idle()

    ui_refs['btn_count'].on_clicked(toggle_count)
    ui_refs['btn_reset'].on_clicked(reset_count)


    # 2. Scores Panel (Center)
    r_score = [0.42, 0.05, 0.18, 0.12]
    ax_scores = reg_ax(r_score, main_view_axes)
    ax_scores.set_xticks([]); ax_scores.set_yticks([])
    ax_scores.set_facecolor('#e8e8e8')
    # Border
    rect_score_border = plt.Rectangle((0,0), 1, 1, transform=ax_scores.transAxes, fill=False, ec='black', lw=2, clip_on=False)
    ax_scores.add_patch(rect_score_border)

    txt_ta_score = ax_scores.text(0.5, 0.75, "INST TA: --", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    val_txt = ax_scores.text(0.5, 0.50, f"TA SET: {GSR_CENTER_VAL:.2f}", ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')
    txt_ta_score.set_animated(True)
    val_txt.set_animated(True)

    # SET Button
    ax_btn_set = reg_ax([0.48, 0.06, 0.06, 0.04], main_view_axes)
    btn_ta_set_now = Button(ax_btn_set, "SET", color='lightblue', hovercolor='cyan')
    ui_refs['btn_ta_set_now'] = btn_ta_set_now
    # [FIX] Set Animated
    btn_ta_set_now.ax.patch.set_animated(True)
    btn_ta_set_now.label.set_animated(True)
    
    def force_set_center(e=None): 
        if latest_gsr_ta > 0.01:
             update_gsr_center(float(f"{latest_gsr_ta:.2f}"))
    btn_ta_set_now.on_clicked(force_set_center)
    
    # === STATUS BAR (Restored) ===
    ax_status = reg_ax([0.05, 0.94, 0.915, 0.04], main_view_axes)
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    ax_status.set_facecolor('#333')
    
    txt_gsr_status = ax_status.text(0.02, 0.5, "GSR: ●", color='lightgray', fontsize=11, fontweight='bold', va='center')
    txt_hrm_status = ax_status.text(0.15, 0.5, "HRM: ●", color='lightgray', fontsize=11, fontweight='bold', va='center')
    
    # [FIX] Init with current Mic Name
    mic_str = "AUDIO: --"
    if 'audio_handler' in globals() and audio_handler.current_mic_name:
         short = audio_handler.current_mic_name
         if "(" in short: short = short.split("(")[0].strip()
         if len(short) > 20: short = short[:20] + "..."
         mic_str = f"AUDIO: {short}"

    txt_audio = ax_status.text(0.67, 0.5, mic_str, color='lightgray', fontsize=11, fontweight='bold', va='center')
    
    # Rec moved left for balance (0.96 -> 0.92)
    rec_text = ax_status.text(0.92, 0.5, "● REC", color='red', fontsize=11, fontweight='bold', va='center', visible=False)
    
    # [FIX] Set Animated
    txt_gsr_status.set_animated(True)
    txt_hrm_status.set_animated(True)
    txt_audio.set_animated(True)
    rec_text.set_animated(True)
    ui_refs['rec_text'] = rec_text
    ui_refs['txt_audio'] = txt_audio # [FIX] Register for UI Callback
    
    ui_refs['txt_gsr_status'] = txt_gsr_status

    # Record Button (Moved Left)
    r_rc = [0.05, 0.12, 0.12, 0.05]
    
    ax_rec = reg_ax(r_rc, main_view_axes)
    ax_rec.set_zorder(1000) # [FIX] Ensure top clickability
    ui_refs['btn_rec'] = Button(ax_rec, "Record", color='#ccffcc', hovercolor='lime')
    ui_refs['btn_rec'].label.set_color('black')
    # [FIX] Animated
    ui_refs['btn_rec'].ax.patch.set_animated(True)
    ui_refs['btn_rec'].label.set_animated(True)
    
    import tkinter as tk

    # [NEW] Globals for Auto-Start Sequence
    pending_rec = False
    pending_notes = ""
    session_start_ta = 0.0

    def start_actual_recording():
        global is_recording, f_gsr, writer_gsr, recording_start_time
        global f_hrm, writer_hrm # [NEW] HRM File
        global notes_filename, audio_filename # [FIX] Restored audio_filename
        # [FIX] Audio Globals Removed (Using AudioHandler)
        global pending_rec, pending_notes, session_start_ta, counting_active, ta_accum # [FIX] Globals
        
        try:
             # Capture Start Stats
             session_start_ta = latest_gsr_ta
             
             # Reset TA Counter
             reset_count(None) # Reset to 0
             
             # Create Files
             DATA_DIR = "Session_Data"
             ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
             SESSION_DIR = os.path.join(DATA_DIR, f"Session_{ts_str}")
             os.makedirs(SESSION_DIR, exist_ok=True)
             
             fname_gsr = os.path.join(SESSION_DIR, "GSR.csv")
             fname_hrm = os.path.join(SESSION_DIR, "HRM.csv") # [NEW]
             audio_filename = os.path.join(SESSION_DIR, "Audio.wav") # [FIX] Define Audio Path

             notes_filename = os.path.join(SESSION_DIR, "notes.txt")
             
             with open(notes_filename, "w") as f:
                 f.write(f"Session Notes - {ts_str}\n")
                 f.write("-" * 30 + "\n")
                 f.write(pending_notes if pending_notes else "No notes provided.")
             
             # Initialize GSR CSV
             f_gsr = open(fname_gsr, 'w', newline='')
             writer_gsr = csv.writer(f_gsr)
             writer_gsr.writerow(["Timestamp", "Elapsed", "TA", "TA SET", "Sensitivity", "Window_Size", "Motion", "Pivot", "Boost", "Notes", "Pattern"])
             
             # (Trend CSV Removed)

             # [NEW] Initialize HRM CSV
             f_hrm = open(fname_hrm, 'w', newline='')
             writer_hrm = csv.writer(f_hrm)
             # Columns: Timestamp, Elapsed, HR_BPM, RMSSD_MS, Raw_RR_MS, State, Trend, Status, Raw_Packet_Hex, Z_HR, Z_HRV, Quadrant
             writer_hrm.writerow(["Timestamp", "Elapsed", "HR_BPM", "RMSSD_MS", "Raw_RR_MS", "State", "Trend", "Status", "Raw_Packet_Hex", "Z_HR", "Z_HRV", "Quadrant"])
 
             is_recording = True 
             # [NEW] Streaming Audio
             audio_handler.start_recording(audio_filename)
             audio_handler.sync_audio_stream(current_view)
             
             recording_start_time = datetime.now()
             
             recording_start_time = datetime.now()
             
             ui_refs['btn_rec'].label.set_text("Stop")
             ui_refs['btn_rec'].color = 'salmon'
             ui_refs['btn_rec'].hovercolor = 'red'
             rec_text.set_visible(True)
             
             # Set Static Info
             s_date = recording_start_time.strftime("%d %B %Y")
             s_time = recording_start_time.strftime("%H:%M")
             if 'txt_sess_date' in ui_refs: ui_refs['txt_sess_date'].set_text(f"Date: {s_date}")
             if 'txt_sess_time' in ui_refs: ui_refs['txt_sess_time'].set_text(f"Time: {s_time}")
             
             log_msg(f"Started: {ts_str}")
             
        except Exception as ex: 
            log_msg(f"Start Err: {ex}")
            is_recording = False

    def toggle_rec(e):
        print("DEBUG: toggle_rec clicked") # [DEBUG]
        global is_recording, f_gsr, writer_gsr, recording_start_time
        global f_hrm, writer_hrm
        global notes_filename, audio_filename
        global pending_rec, pending_notes, calib_mode, calib_step, calib_phase, calib_start_time, calib_base_ta, calib_min_ta, counting_active
        
        if not is_recording:
             if audio_handler.selected_device_idx is None:
                  print("DEBUG: No Mic Selected") # [DEBUG]
                  log_msg("Err: No Mic Selected!")
                  return
                 
             root = tk.Tk(); root.withdraw()
             note_data = {"text": None}
             
             dlg = tk.Toplevel(root)
             dlg.title("Session Notes")
             dlg.geometry("500x400")
             
             tk.Label(dlg, text="Enter Session Details:", font=("Arial", 10, "bold")).pack(pady=5)
             template_text = "Client Name: \n\nProcess Run: \n\nOther Notes:"
             
             txt = tk.Text(dlg, width=50, height=15, font=("Arial", 10))
             txt.pack(padx=10, pady=5, expand=True, fill='both')
             txt.insert("1.0", template_text)
             txt.focus_set()
             
             def on_submit():
                 note_data["text"] = txt.get("1.0", "end-1c")
                 dlg.destroy()
                 
             def on_cancel():
                 dlg.destroy()
                 
             btn_frame = tk.Frame(dlg)
             btn_frame.pack(pady=10)
             tk.Button(btn_frame, text="Start Recording", command=on_submit, bg="#ddffdd", height=2).pack(side='left', padx=10)
             tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side='left', padx=10)
             
             root.wait_window(dlg)
             root.destroy()
             
             if note_data["text"] is None: return # User Cancelled
             
             # [NEW] Queue Calibration
             pending_notes = note_data["text"]
             start_actual_recording() # [REQ] Start timer immediately
             pending_rec = True # Use this flag to trigger the "SESSION STARTED" message after calib
             
             # Start Calibration via centralized function (Ensures Boost handling)
             start_calibration(None)
             
        else:
             is_recording = False
             audio_handler.is_recording = False
             if counting_active: toggle_count(None) # [REQ] Stop TA Counter on Session Stop
             
             # If stopping during calibration?
             if calib_mode:
                 calib_mode = False
                 pending_rec = False
                 log_msg("Start Aborted.")
                 return # Exit early, no session file to close?
              
             ui_refs['btn_rec'].label.set_text("Record")
             ui_refs['btn_rec'].color = '#ccffcc'
             ui_refs['btn_rec'].hovercolor = 'lime'
             try: 
                 # fig.canvas.draw_idle() 
                 pass
             except: pass # [FIX] Non-blocking refresh 
             
             if f_gsr: f_gsr.close()
             if f_hrm: f_hrm.close() # [NEW]
              
             audio_handler.stop_recording()
             audio_handler.sync_audio_stream(current_view) 
             
             # [NEW] Append Summary to Notes
             try:
                 if recording_start_time and notes_filename:
                     end_time = datetime.now()
                     dur = end_time - recording_start_time
                     total_sec = int(dur.total_seconds())
                     hours, remainder = divmod(total_sec, 3600)
                     mins, secs = divmod(remainder, 60)
                     final_len_str = f"{hours:02}:{mins:02}:{secs:02}"
                     
                     start_fmt = recording_start_time.strftime("%H:%M:%S")
                     
                     # Stats
                     end_ta = latest_gsr_ta
                     total_count = ta_accum
                     
                     with open(notes_filename, "a") as f:
                         f.write(f"\n\n{'='*30}\n")
                         f.write("SESSION SUMMARY\n")
                         f.write(f"Start Time: {start_fmt}\n")
                         f.write(f"Duration  : {final_len_str}\n")
                         f.write(f"Start TA  : {session_start_ta:.2f}\n")
                         f.write(f"End TA    : {end_ta:.2f}\n")
                         f.write(f"TA Count  : {total_count:.2f}\n")
                     log_msg(f"Notes Updated: Duration {final_len_str}, Count {total_count:.2f}")
             except Exception as ex:
                 log_msg(f"Note Update Err: {ex}")
             rec_text.set_visible(False)
             log_msg("Recording Saved")

    ui_refs['btn_rec'].on_clicked(toggle_rec)
    
    # 3. Session Details Panel (Right of Scores) (RESTORED v35 Position)
    r_detail = [0.62, 0.05, 0.20, 0.12]
    ax_detail = reg_ax(r_detail, main_view_axes)
    ax_detail.set_xticks([]); ax_detail.set_yticks([])
    ax_detail.set_facecolor('#f9f9f9')
    rect_det_border = plt.Rectangle((0,0), 1, 1, transform=ax_detail.transAxes, fill=False, ec='#aaaaaa', lw=2, clip_on=False)
    ax_detail.add_patch(rect_det_border)
    
    # Title
    ax_detail.text(0.5, 0.85, "Session Detail", ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    
    # Text Fields
    txt_sess_date = ax_detail.text(0.05, 0.60, "Date: --", ha='left', va='center', fontsize=9, color='#333')
    txt_sess_time = ax_detail.text(0.05, 0.40, "Time: --", ha='left', va='center', fontsize=9, color='#333')
    txt_sess_len  = ax_detail.text(0.05, 0.15, "Duration : 00:00:00", ha='left', va='center', fontsize=10, fontweight='bold', color='black')
    txt_sess_len.set_animated(True)
    
    ui_refs['txt_sess_date'] = txt_sess_date
    ui_refs['txt_sess_time'] = txt_sess_time
    ui_refs['txt_sess_len'] = txt_sess_len

    # [NEW] Vitals Display (HR/HRV) - Restored to spare space
    txt_hr_val = ax_detail.text(0.55, 0.60, "HR: --", ha='left', va='center', fontsize=10, fontweight='bold', color='red')
    txt_hrv_val = ax_detail.text(0.55, 0.40, "HRV: --", ha='left', va='center', fontsize=10, fontweight='bold', color='blue')
    txt_hr_val.set_animated(True)
    txt_hrv_val.set_animated(True)
    ui_refs['txt_hr_val'] = txt_hr_val
    ui_refs['txt_hrv_val'] = txt_hrv_val

    r_ts = [0.05, 0.06, 0.12, 0.05]
    ax_to_set = reg_ax(r_ts, main_view_axes)
    ui_refs['btn_to_settings'] = Button(ax_to_set, "Settings >", color='lightgray')
    ui_refs['btn_to_settings'].ax.patch.set_animated(True)
    ui_refs['btn_to_settings'].label.set_animated(True)

    # --- SETTINGS PAGE ELEMENTS ---
    # [FIX] Full Screen Blind to cover Main Page artifacts during Blitting
    ax_blind = reg_ax([0,0,1,1], settings_view_axes)
    ax_blind.set_xticks([]); ax_blind.set_yticks([])
    ax_blind.set_facecolor('white') 
    ax_blind.set_zorder(-5) 
    
    rect_audio = [0.35, 0.28, 0.28, 0.25] 
    ax_audio_bg = create_panel_ax(rect_audio, "Audio Input Control")
    ui_refs['ax_audio_bg'] = ax_audio_bg
    
    # 3. Audio Input Control
    ax_mic_lbl = reg_ax([rect_audio[0]+0.02, rect_audio[1]+0.16, rect_audio[2]-0.04, 0.06], settings_view_axes)
    ax_mic_lbl.axis('off')
    ui_refs['text_mic_name'] = ax_mic_lbl.text(0, 0.5, "NO MIC", va="center", ha="left", fontsize=9)
    
    r_meter = [rect_audio[0]+0.02, rect_audio[1]+0.11, rect_audio[2]-0.09, 0.02] 
    ax_level = reg_ax(r_meter, settings_view_axes)
    ax_level.set_xlim(0, 1.0); ax_level.set_ylim(-0.5, 0.5) 
    ax_level.set_xticks([]); ax_level.set_yticks([])
    ax_level.set_facecolor("#333")
    ax_level.set_zorder(10) 
    bar_level = ax_level.barh([0], [0], color='green', height=1.0)
    
    ax_lvl_txt = reg_ax([rect_audio[0]+rect_audio[2]-0.06, rect_audio[1]+0.10, 0.05, 0.04], settings_view_axes)
    ax_lvl_txt.axis('off')
    ui_refs['text_level'] = ax_lvl_txt.text(0.5, 0.5, "0%", ha='center', va='center', fontsize=8, color='blue')

    r_msel = [rect_audio[0]+0.02, rect_audio[1]+0.06, 0.20, 0.035]
    ax_msel = reg_ax(r_msel, settings_view_axes)
    ax_msel.set_zorder(20) # [FIX] Ensure visible
    ui_refs['btn_select_mic'] = Button(ax_msel, 'Select Input...', color='lightyellow')
    ui_refs['btn_select_mic'].on_clicked(lambda e: audio_handler.open_audio_select())

    r_gain = [rect_audio[0] + 0.02, rect_audio[1] + 0.01, rect_audio[2] - 0.04, 0.03]
    ax_gain = reg_ax(r_gain, settings_view_axes)
    ui_refs['slide_gain'] = Slider(ax_gain, 'Mic Gain', 1.0, 10.0, valinit=audio_handler.current_mic_gain, color='lime')
    ui_refs['slide_gain'].on_changed(lambda v: setattr(audio_handler, 'current_mic_gain', v))


    
    # Back Button
    r_bk = [0.05, 0.90, 0.10, 0.04]
    ax_back = reg_ax(r_bk, settings_view_axes)
    ui_refs['btn_back'] = Button(ax_back, "< Back", color='lightgray')
    # [FIX] Do NOT animate button back, as Settings View uses standard draw()
    # ui_refs['btn_back'].ax.patch.set_animated(True)
    # ui_refs['btn_back'].label.set_animated(True)
    
    # SYSTEM INFO
    r_info = [0.0, 0.0, 1.0, 0.04] 
    ax_info = reg_ax(r_info, main_view_axes) # [FIX] Move to Main View
    ax_info.set_axis_off()
    system_line = ax_info.text(0.5, 0.5, "Waiting for Info...", ha="center", family=['Arial', 'Noto Emoji', 'sans-serif'])
    system_line.set_animated(True) # [FIX] For Manual Blit
    ui_refs['system_line'] = system_line # Store ref
    
    def teleport_off(ax_list):
        for a in ax_list: a.set_visible(False); a.set_position([1.5, 1.5, 0.01, 0.01])

    def teleport_on(ax_list):
        for a in ax_list: a.set_visible(True); a.set_position(ax_positions[a])

    teleport_off(settings_view_axes)
  
    def req_main(e): 
        global desired_view; desired_view = 'main'
        audio_handler.sync_audio_stream('main')
        # fig.canvas.draw() # [FIX] Removed from here, moved to update() switch logic
 # [FIX] Force clear for View Switch (Blit)
    def req_settings(e): 
        global desired_view; desired_view = 'settings'
        audio_handler.sync_audio_stream('settings')
        fig.canvas.draw() # [FIX] Force clear for View Switch (Blit)
    
    ui_refs['btn_to_settings'].on_clicked(req_settings)
    ui_refs['btn_back'].on_clicked(req_main)
    
    dev_static = {}
    
    # [NEW] Helper Functions (Global Scope)
    def force_set_center(e=None):
        # if is_recording or not is_connected: return # Optional constraint
        
        # [REQ] Apply pending Auto-Boost Zoom on Reset
        global active_boost_level
        active_boost_level = booster_level
        
        # Force center
        update_gsr_center(latest_gsr_ta)
        #plt.draw()
        
    
    def update(frame=0):
        global current_view, desired_view, current_state, event_detected, ignore_ui_callbacks
        global last_motion_log, motion_lock_expiry, motion_prev_ta, motion_prev_time
        global headset_on_head, BASE_SENSITIVITY, saved_boost_level, booster_level
        global calib_mode, calib_phase, calib_step, calib_start_time, calib_step_start_time, calib_base_ta, calib_min_ta, calib_vals, last_calib_ratio
        global recording_start_time, is_recording, session_start_ta, pending_rec
        global latest_hr, latest_hrv, hrm_status, active_event_label, is_connected_prev, prev_grid_state
        global first_run_center # [FIX] Add global
        
        # [NEW] Manual Blit Restore (Main View Only)
        if current_view == 'main':
            if graph_bg: fig.canvas.restore_region(graph_bg)
            if bg_scores: fig.canvas.restore_region(bg_scores)
            if bg_count:  fig.canvas.restore_region(bg_count)
            if bg_detail: fig.canvas.restore_region(bg_detail)
            if bg_status: fig.canvas.restore_region(bg_status)
            if bg_sens:   fig.canvas.restore_region(bg_sens)
            if bg_info:   fig.canvas.restore_region(bg_info) # [FIX] Restore Info

        if not first_run_center and latest_gsr_ta > 0.1:
             print(f"[Auto-Center] First Reading: {latest_gsr_ta}")
             update_gsr_center(latest_gsr_ta)
             first_run_center = True
        
        current_state = "Audio/GSR Mode"
        
        ax_graph.grid(True, alpha=0.3)
        ax_graph.get_yaxis().set_visible(False)
        
        if is_recording: rec_text.set_alpha(1.0)
        else: rec_text.set_alpha(0.0)
        
        if current_view != desired_view:
            if desired_view == 'settings': 
                teleport_off(main_view_axes); teleport_on(settings_view_axes)
            elif desired_view == 'main': 
                teleport_off(settings_view_axes); teleport_on(main_view_axes)
                # [FIX] Force clean draw to clear Settings artifacts from buffer
                fig.canvas.draw()
                
            current_view = desired_view

        while not ui_update_queue.empty():
            info = ui_update_queue.get_nowait()
            dev_static.update(info)
            if 'name' in info:
                 try:
                     ignore_ui_callbacks = True
                     ui_refs['text_name'].set_val(info['name'])
                 except: pass
                 finally: ignore_ui_callbacks = False
        
        # [NEW] Poll HRM before System Line
        h_stat = "Disabled"
        h_batt_str = ""
        s_state = "Init"
        s_trend = "-"
        s_intens = "-"
        
        if hrm_sensor:
            # Poll data
            h_data = hrm_sensor.get_data()
            latest_hr = h_data['bpm']
            latest_hrv = h_data['rmssd']
            h_stat = h_data['status']
            
            # [NEW] HRM Battery Info
            batt_v = h_data.get('battery_volts')
            batt_st = h_data.get('battery_state', 'Unknown')
            if batt_v:
                h_batt_str = f" | 🔋 {batt_v}V ({batt_st})"
            
            # [NEW] Analyze State
            analyzer_res = {}
            if hrv_analyzer and h_stat == "Active":
                analyzer_res = hrv_analyzer.update(latest_hr, latest_hrv)
                latest_hrm_state = analyzer_res

            s_state = analyzer_res.get('state', 'Init')
            s_trend = analyzer_res.get('trend', '-')
            s_intens = analyzer_res.get('intensity', '-')
            
            # [NEW] Balanced State Logic
            if s_intens == "Low":
                s_state = "BALANCED"

            # [NEW] Update Grid
            if hrv_analyzer:
                z_h = analyzer_res.get('z_hr', 0)
                z_v = analyzer_res.get('z_hrv', 0)
                if 'update_grid' in locals() or 'update_grid' in globals():
                    update_grid(z_h, z_v, s_state, s_trend)
        # System Line
        sys_parts = []
        sys_parts.append(f"Mic: {audio_handler.current_mic_name}")
        
        if h_stat == "Active":
             sys_parts.append(f"HRM: {h_stat}{h_batt_str}")
             
        s_str = " | ".join(sys_parts)
        system_line.set_text(s_str)
        
        # [NEW] Update HRM Top Status Bar (Minimal Dot)
        if hrm_sensor:
            if h_stat == "Active":
                 txt_hrm_status.set_color('#009900')
                 txt_hrm_status.set_text(f"HRM: ●") 
            else:
                 # Color logic for non-active states
                 if h_stat == "Initializing": 
                     txt_hrm_status.set_color('lightgray')
                 elif h_stat == "Signal Lost":
                     txt_hrm_status.set_color('orange')
                 elif "Error" in h_stat:
                     txt_hrm_status.set_color('red')
                 else:
                     txt_hrm_status.set_color('yellow')
                 
                 # Always show Dot as requested
                 txt_hrm_status.set_text("HRM: ●")
            
            # Update Detail Panel
            if 'txt_hr_val' in ui_refs:
                 ui_refs['txt_hr_val'].set_text(f"HR: {latest_hr}")
                 # [Clean] Removed Trend/Intensity Text
                 ui_refs['txt_hrv_val'].set_text(f"HRV: {int(latest_hrv)}ms")

            # [NEW] Log to HRM CSV if Recording
            if is_recording and hrm_sensor:
                try:
                    ts_hrm = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    rr_val = h_data.get('rr_ms', 0)
                    raw_rr = h_data.get('raw_rr_ms', 0)
                    raw_hex = h_data.get('raw_hex', '')
                    
                    if writer_hrm and h_stat == "Active":
                        # Timestamp, HR_BPM, RMSSD_MS, Raw_RR_MS, State, Trend, Status, Raw_Packet_Hex
                        # Use Analyzer Output for State and Trend
                        log_state = f"{s_state}:{s_intens}"
                        z_h_log = analyzer_res.get('z_hr', '0.000')
                        z_v_log = analyzer_res.get('z_hrv', '0.000')
                        quad_log = analyzer_res.get('quadrant', 0)
                        
                        elapsed = 0.0
                        if recording_start_time:
                             elapsed = (datetime.now() - recording_start_time).total_seconds()
                        
                        writer_hrm.writerow([ts_hrm, f"{elapsed:.3f}", latest_hr, int(latest_hrv), raw_rr, log_state, s_trend, h_stat, raw_hex, z_h_log, z_v_log, quad_log])
                except: pass

        else:
            txt_hrm_status.set_color('gray')
            
        # [Moved] Update GSR UI
        # 1. Update Text
        txt_ta_score.set_text(f"INST TA: {latest_gsr_ta:.3f}")
        txt_count_val.set_text(f"TA Counter: {ta_accum:.2f}") # [NEW] Update Counter
        
        # [NEW] Check Motion Status (Anti-Squirm/Velocity)
        # Calculate Speed (TA/sec)
        now = time.time()
        if motion_prev_time == 0.0: 
             motion_prev_time = now
             motion_prev_ta = latest_gsr_ta
        
        dt = now - motion_prev_time
        velocity = 0.0
        if dt > 0.0:
             velocity = abs(latest_gsr_ta - motion_prev_ta) / dt
             # [REQ] Hardened: 1.5 TA/s Threshold, 2.0s Duration
             if velocity > 1.5:
                  motion_lock_expiry = now + 2.0 # Lock for 2s
                  # [REQ] Log Once per engagement
                  global motion_lock_active
                  if not motion_lock_active:
                       active_event_label = "MOTION_LOCK_ENGAGED"
                       motion_lock_active = True
                  
             # [NEW] Manage Motion Overlay & State
             overlay_text = ""
             if not calib_mode: # [FIX] Do not overwrite Calibration Overlay
                 if now < motion_lock_expiry:
                     overlay_text = f"MOTION (Vel={velocity:.1f})"
                     if 'txt_motion_overlay' in ui_refs:
                         ui_refs['txt_motion_overlay'].set_text(overlay_text)
                 else:
                      if 'txt_motion_overlay' in ui_refs:
                          ui_refs['txt_motion_overlay'].set_text("")
                      motion_lock_active = False # Reset state
                      
        motion_prev_time = now
        motion_prev_ta = latest_gsr_ta

        # Calculate is_moving and is_steady for Logic Blocks
        is_moving = (time.time() < motion_lock_expiry)
        
        # Calculate Steadiness
        history = list(bands_history["GSR"])[-60:]
        is_steady = True
        rng = 0.0
        if len(history) > 30:
             rng = max(history) - min(history)
             if rng > 0.2: is_steady = False

        # [NEW] Update GSR Status Label (Light)
        # [FIX] Throttled Text Updates
        # [FIX] Throttled Text Updates
        # [FIX] Always calculate effective window (used every frame)
        eff_win = get_effective_window()

        if True: # Always update text in Manual Blit mode (it's fast enough)
             # Scale/Sens Text
             eff_win = get_effective_window()
             disp_sens = 99.9 if eff_win <= 0.001 else (1.0 / eff_win)
             txt_win_val.set_text(f"{disp_sens:.2f}")

             # TA SET Label
             if 'txt_ta_set_line' in ui_refs:
                  ui_refs['txt_ta_set_line'].set_text(f"TA SET: {GSR_CENTER_VAL:.2f}")

             # Rec Status (Blink)
             # Use time, not frame count
             if is_recording: 
                  # blink 1s on 1s off
                  rec_text.set_alpha(1.0 if int(time.time()*2) % 2 == 0 else 0.0)
             else: rec_text.set_alpha(0.0)

             # GSR Status Label
             if 'txt_gsr_status' in ui_refs:
                  if gsr_thread and gsr_thread.connected:
                       ui_refs['txt_gsr_status'].set_text("GSR: ●")
                       ui_refs['txt_gsr_status'].set_color('#009900')
                  else:
                       ui_refs['txt_gsr_status'].set_color('red')
             
        # [ALWAYS RUN] Graph Line Logic below...
        if latest_gsr_ta > 0.01 and booster_level != 0: # Only if valid signal and Boost ON
            # Recalculate limits for check
            min_p = GSR_CENTER_VAL - (0.625 * eff_win)
            max_p = GSR_CENTER_VAL + (0.375 * eff_win)
            
            if latest_gsr_ta < min_p or latest_gsr_ta > max_p:
                if is_steady and not is_moving:
                    # Auto-Center
                    # [REQ] Apply pending Auto-Boost Zoom on Auto-Reset
                    active_boost_level = booster_level
                    r_type = "Fall" if latest_gsr_ta < min_p else "Rise"
                    update_gsr_center(latest_gsr_ta, reason=f"Auto-Reset (Boost-{r_type})")
                else:
                     # Log blocked boost reset
                     if time.time() - last_motion_log > 2.0:
                         log_msg(f"Motion Detected (Vel={velocity:.1f}, Rng={rng:.2f}) -> Auto-Reset (Boost) BLOCKED")
                         last_motion_log = time.time()
            
        if current_view == 'main':
             # [REQ] Update Pattern Engine
             global current_pattern, gsr_patterns
             if gsr_patterns is None: gsr_patterns = GSRPatterns(history_len_sec=8.0)
             
             # Only update if we have data
             now = time.time() # [FIX] Ensure 'now' is defined
             if latest_gsr_ta > 0.1:
                  eff_win = get_effective_window()
                  
                  # [REQ] Check Motion Status
                  is_moving = (time.time() < motion_lock_expiry)
                  
                  raw_pat = gsr_patterns.update(latest_gsr_ta, now, effective_window=eff_win, is_motion=is_moving)
                  
                  # [REQ] PATTERN HOLD LOGIC (Debounce)
                  # If we hit a "Transient" read (Tick, Rocket, Slam), hold it for 1.2s
                  # pattern_hold_until must be global
                  global pattern_hold_until
                  if 'pattern_hold_until' not in globals(): pattern_hold_until = 0.0
                  
                  # [MOD] Removed TICK/STAGE FOUR from transients list
                  transients = ["ROCKET READ", "BLOWDOWN", "SHORT FALL", "SHORT RISE"]
                  
                  if raw_pat in transients:
                       current_pattern = raw_pat
                       pattern_hold_until = now + 1.2 # Hold for 1.2s
                  else:
                       # Only switch to Background state (Fall, Rise, Idle) if Hold expired
                       if now > pattern_hold_until:
                            current_pattern = raw_pat
                       # Else: Keep showing the held pattern
                  
                  pat = current_pattern # For color logic below
                  
                  if 'txt_pattern' in ui_refs:
                       # Color logic
                       col = 'gray'
                       if pat == "BLOWDOWN": col = '#006400'
                       elif pat == "ROCKET READ": col = '#DC143C'  # Crimson
                       elif pat == "LONG FALL": col = '#008000'
                       elif pat == "SHORT FALL": col = '#3CB371'
                       elif pat == "LONG RISE": col = '#FF4500'    # OrangeRed
                       elif pat == "SHORT RISE": col = 'orange'
                       elif pat == "STUCK": col = 'red'
                       elif pat == "MOTION": col = '#8B0000'       # Dark Red (Motion Locked)
                       
                       disp_pat = f"PATTERN: {pat}" if pat else ""
                       ui_refs['txt_pattern'].set_text(disp_pat)
                       ui_refs['txt_pattern'].set_color(col)
                       
             
             # [NEW] Check for Auto-Start Recording (Triggered by 'R' key in background?)
             # Implement User Keybinds? 
             pass   # [OPT] Only update visible lines.
             
             for k, line in lines.items():
                 if not line.get_visible(): continue
                                 
                 # [REQ] Skip GSR if not connected
                 if k == 'GSR':
                      gsr_ok = False
                      if 'gsr_thread' in globals() and gsr_thread:
                           if gsr_thread.connected: gsr_ok = True
                      
                      if not gsr_ok:
                           if len(line.get_xdata()) > 0: line.set_data([], [])
                           continue
                 
                 data = bands_history[k]
                 if len(data) > 0:
                     # [OPT] Downsample (Decimation)
                     # [FIX] Thread-Safety: list(data) snapshots the deque to prevent mutation during iteration
                     # Using stride of 2
                     raw_full = list(data)
                     raw_ys = np.array(raw_full[::2])
                     
                     if k == 'GSR':
                          # [FIX] Vectorized Dynamic Normalization (Speedup)
                          min_p = GSR_CENTER_VAL - (0.625 * eff_win)
                          max_p = GSR_CENTER_VAL + (0.375 * eff_win)
                          span_p = max_p - min_p if (max_p - min_p) > 0.001 else 0.001
                          
                          # Vectorized Math
                          ys = (raw_ys - min_p) / span_p * 100.0
                     else:
                         ys = raw_ys

                     # [FIX] Scale X-axis by stride (2)
                     line.set_data(np.arange(len(ys)) * 2, ys)
                 else:
                     line.set_data([], [])

        if audio_handler.audio_stream and audio_handler.audio_stream.active and current_view == 'settings':
             raw_lvl = audio_handler.audio_state['peak']
             lvl = min(1.0, raw_lvl) 
             
             bar_level[0].set_width(lvl)
             ui_refs['text_level'].set_text(f"{int(lvl*100)}%") 
             # Button State Update (Standard)
            # This block was removed as per instruction.
        else:
             bar_level[0].set_width(0)
             ui_refs['text_level'].set_text("OFF")

        # [NEW] Add Auto-Center Check to end of loop
        if not calib_mode and current_view == 'main':
             # [FIX] global motion_lock_expiry moved to top
             
             eff_win = get_effective_window()
             min_p = GSR_CENTER_VAL - (0.625 * eff_win)
             max_p = GSR_CENTER_VAL + (0.375 * eff_win)
             
             if latest_gsr_ta < min_p or latest_gsr_ta > max_p:
                 # Only if valid signal
                 if latest_gsr_ta > 0.1:
                      # [REQ] Motion Lock (Anti-Squirm)
                      # If signal is jumping (Range > 0.2 over last 1s), DO NOT Rescale/Recenter.
                      # This prevents "crazy" sensitivity during motion artifacts.
                      if time.time() > motion_lock_expiry and is_steady:
                            if latest_gsr_ta < min_p: 
                                active_event_label = "DROP_TA_RESET"
                                reason_str = "Auto-Reset (Fall)"
                            else: 
                                active_event_label = "RISE_TA_RESET"
                                reason_str = "Auto-Reset (Rise)"
                            update_gsr_center(latest_gsr_ta, reason=reason_str)
                      else:
                           # [REQ] Log Motion Block (Throttled)
                           
                           if time.time() - last_motion_log > 2.0:
                               log_msg(f"Motion Detected (Vel={velocity:.1f}, Rng={rng:.2f}) -> Auto-Reset BLOCKED")
                               last_motion_log = time.time()
                               active_event_label = "MOTION_BLOCKING_RESET"

        if current_view == 'main':
             pass

             if calib_mode:
                try:
                    # Ensure overlay access
                    if 'txt_calib_overlay' in ui_refs:
                        ovl = ui_refs['txt_calib_overlay']
                        ovl.set_visible(True)
                    else:
                        ovl = None
                        print("Error: Overlay not in ui_refs")

                    elapsed = time.time() - calib_start_time
                    
                    # --- Steps 1, 2, 3: SQUEEZE ---
                    if calib_step in [1, 2, 3]:
                        if calib_phase == 0: # WAIT FOR DROP

                             history = list(bands_history['GSR'])[-60:]
                             is_steady = False
                             rng = 1.0
                             if len(history) > 30:
                                 rng = max(history) - min(history)
                                 # [MOD] Relaxed threshold from 0.05 to 0.15 to allow for settling
                                 if rng < 0.15: is_steady = True
                             
                             # [FIX] Bypass Steadiness Check if we already have a Large Drop (Early Squeeze)
                             has_large_drop = (calib_base_ta - latest_gsr_ta) > 0.05
                             
                             if elapsed < 2.0 or (not is_steady and not has_large_drop):
                                  msg = f"CALIB {calib_step}/4: RELAX HAND..."
                                  
                                  # [REQ] Hysteresis Tracking:
                                  # 1. Follow Rises (Recovery)
                                  if latest_gsr_ta > calib_base_ta:
                                      calib_base_ta = latest_gsr_ta
                                  else:
                                      # 2. Follow TINY Drops (Michron-Drift) only.
                                      # If drop > 0.002, we assume it might be a start of a squeeze, so HOLD Base.
                                      if (calib_base_ta - latest_gsr_ta) < 0.002:
                                          calib_base_ta = latest_gsr_ta
                                  # Else: We Hold 'calib_base_ta' at the peak, assuming this is a Squeeze.
                             else:
                                  # [NEW] Log Gate Open for Analysis
                                  if calib_phase == 0: # One-time log on transition
                                      # We don't want to flood log, but this block runs every frame until phase changes.
                                      # Actually phase changes below immediately.
                                      pass
                                      
                                  msg = f"CALIB {calib_step}/4: SQUEEZE SENSOR\n(WAITING FOR DROP)"
                                  if latest_gsr_ta < calib_base_ta - 0.05: # Trigger Drop (Hardened from 0.02)
                                      active_event_label = f"RELAX_GATE_OPEN - Rng:{rng:.3f}" # Log the state we came from
                                      calib_phase = 1
                                      calib_min_ta = latest_gsr_ta
                                      calib_start_time = time.time() # Reset timer for safety
                                      active_event_label = f"SQUEEZE_{calib_step}_DROP"
                        elif calib_phase == 1: # TRACKING DROP
                             msg = f"CALIB {calib_step}/4: AND RELEASE..."
                             if latest_gsr_ta < calib_min_ta:
                                 calib_min_ta = latest_gsr_ta
                             
                             # Now we only update sensitivity after stabilization (Phase 3).
                             
                             # Detect Release (75% Recovery)
                             current_drop = calib_base_ta - calib_min_ta
                             # If drop is tiny (<0.05), use fixed 0.05 rise. Else use 75% return
                             recovery_target = calib_min_ta + (current_drop * 0.75) if current_drop > 0.05 else calib_min_ta + 0.05
                             
                             if latest_gsr_ta > recovery_target:
                                 calib_phase = 3
                                 calib_start_time = time.time() # Start stability timer
                                 active_event_label = f"SQUEEZE_{calib_step}_RELEASE"
                                 
                        elif calib_phase == 3: # STABILIZATION
                             msg = "CALIB: STABILIZING..."
                             # Simple wait 1.5s for "Level Off"
                             # Ideally check variance, but 1.5s post-release is usually enough
                             if time.time() - calib_start_time > 1.5:
                                 calib_phase = 2 # Go to Calculation (Phase 2 became Calculate?)
                                 # Re-route: I should just do calculation here or change Phase 2 to 'Calculate'
                                 
                                 # Let's promote Phase 3 to "Finishing Step" logic
                                 # Calculate drop
                                 total_drop = calib_base_ta - calib_min_ta
                                 if total_drop < 0.05: total_drop = 0.05
                                 calib_vals.append(total_drop)
                                 
                                 log_msg(f"Calib {calib_step}: Collected Drop={total_drop:.4f}")
                                 active_event_label = f"SQUEEZE_{calib_step}_COLLECTED"
                                 
                                 # [REQ] Median Logic at End of Step 3
                                 if calib_step == 3:
                                      # Compute Median
                                      median_drop = sorted(calib_vals)[1] # Middle of 3
                                      target_win = median_drop / CALIB_TARGET_RATIO
                                      target_win = max(0.05, min(50.0, target_win))
                                      
                                      BASE_SENSITIVITY = target_win
                                      log_msg(f"Calibration Complete: Median Drop={median_drop:.4f} -> Sensitivity={1.0/BASE_SENSITIVITY:.2f}")
                                      active_event_label = "CALIB_COMPLETE_MEDIAN"
                                      
                                      # Update Display
                                      update_gsr_center(latest_gsr_ta, reason="Calib (Final)")

                                 # [REQ] Reset TA (Center Graph) after sensitivity change
                                 update_gsr_center(latest_gsr_ta, reason="Calib (Resized)")
                             
                                 # Move Next
                                 calib_step += 1
                                 calib_phase = 0
                                 calib_start_time = time.time()
                                 calib_base_ta = latest_gsr_ta 
                                 calib_min_ta = latest_gsr_ta
                                 

                        if ovl: ovl.set_text(msg)
                            
                            
                    # --- Step 4: BREATH ---
                    elif calib_step == 4:
                        if calib_phase == 0:
                            # [NEW] Timeout Check (30s)
                            if time.time() - calib_start_time > 30.0:
                                 log_msg("Calib Step 4 Timeout - Restarting")
                                 msg = "TIMEOUT - RESTARTING..."
                                 if ovl: ovl.set_text(msg) # Show brief err
                                 start_calibration(None) # Restart
                                 return # Exit current loop
                            
                            # [REQ] Stability Gate
                            # Bypass Steadiness if Large Drop
                            has_large_drop = (calib_base_ta - latest_gsr_ta) > 0.05
                            

                            if time.time() - calib_start_time < 2.0:
                                 msg = "CALIB 4/4: RELAX HAND..."
                                 
                                 # [REQ] Hysteresis Tracking (Same as Squeeze)
                                 if latest_gsr_ta > calib_base_ta:
                                     calib_base_ta = latest_gsr_ta
                                 else:
                                     if (calib_base_ta - latest_gsr_ta) < 0.05:
                                         calib_base_ta = latest_gsr_ta
                                 # Else: Peak Hold
                            else:
                                 msg = "CALIB 4/4: DEEP BREATH IN\n(WAITING FOR DROP)"
                                 if latest_gsr_ta < calib_base_ta - 0.02:
                                    calib_phase = 1
                                    calib_min_ta = latest_gsr_ta
                                    calib_start_time = time.time() # [REQ] Start Stability Timer
                                    active_event_label = "BREATH_DROP"
                                    calib_step_start_time = time.time() # [NEW] Track total duration of this phase
                        elif calib_phase == 1:
                            msg = "CALIB 4/4: AND RELEASE..."
                            
                            # Track Minimum (Deepest Point)
                            if latest_gsr_ta < calib_min_ta:
                                calib_min_ta = latest_gsr_ta
                                calib_start_time = time.time() # Reset Stability Timer on new low
                            
                            # Check Stability (No new low for 1.5s) AND Min Duration (4.0s)
                            # This prevents failing immediately if the user pauses or starts slowly.
                            time_in_phase = time.time() - calib_step_start_time
                            is_stable = (time.time() - calib_start_time > 1.5)
                            
                            if is_stable and time_in_phase > 4.0:
                                # Validate Drop
                                total_drop = calib_base_ta - calib_min_ta
                                ratio = total_drop / BASE_SENSITIVITY
                                last_calib_ratio = ratio # [DEBUG]
                                
                                # [FIX] Strict Range Removed. Any valid drop is OK.
                                # Drop > 1.0 means off-screen (FAIL) but rare.
                                if False: # Disabled Check
                                    pass
                                else:
                                    # Success
                                    calib_phase = 2
                                    calib_start_time = time.time()                                    
                                    update_gsr_center(latest_gsr_ta, reason="Calib (Breath)")
                                    active_event_label = f"BREATH_VERIFIED | Drop={total_drop:.4f}"
                                
                        elif calib_phase == 2:
                            msg = "CALIB: COMPLETE!"
                            if time.time() - calib_start_time > 2.0: # Show Success for 2s
                                # [REQ] Restore previous Auto-Boost Level (Primed only)
                                log_msg(f"Restoring Boost: {saved_boost_level} (Primed)")
                                # Do NOT call set_boost() as it might force immediate update in some versions or be safe.
                                # Safest to just set var and update UI.
                                booster_level = saved_boost_level
                                update_boost_ui()
                                # active_boost_level remains 0 (from Start Calibration) until next Reset.
                                
                                # [NEW] Auto-Start Recording if Pending
                                if pending_rec:
                                    pending_rec = False
                                    if not is_recording: start_actual_recording()
                                    # [REQ] Show "SESSION STARTED"
                                    # Enter Phase 3 just to show this message
                                    calib_phase = 3
                                    calib_start_time = time.time()
                                else:
                                    # If not recording, we are done
                                    if is_recording:
                                        reset_count(None)
                                        session_start_ta = latest_gsr_ta
                                        
                                    calib_mode = False
                                    update_gsr_center(latest_gsr_ta, force_pivot=True, reason="Calib (Complete)") # [REQ] Force Pivot Update
                                    if ovl: ovl.set_text("")
                                    log_msg("Calibration Complete")
                                    active_event_label = "CALIB_COMPLETE"

                        elif calib_phase == 3:
                             msg = "SESSION STARTED"
                             if not counting_active: toggle_count(None)
                             
                             if time.time() - calib_start_time > 2.0:
                                  calib_mode = False
                                  update_gsr_center(latest_gsr_ta, force_pivot=True) # [REQ] Force Pivot Update
                                  # if gsr_patterns: gsr_patterns.reset() # [FIX] Disabled
                                  active_event_label = "SESSION_STARTED" # [REQ] Log Event
                                  if ovl: ovl.set_text("")
                                  log_msg("Calibration Sequence Finished")

                        elif calib_phase == 5:
                             # [NEW] Error Phase
                             msg = f"CALIBRATION FAILED\nRATIO: {last_calib_ratio:.2f} (0.30 - 0.65)"
                             active_event_label = f"CALIB_FAILED - RATIO: {last_calib_ratio:.2f}" # Drop not avail here easily without saving
                             if time.time() - calib_start_time > 2.0:
                                 # Restart
                                 calib_step = 1
                                 calib_phase = 0
                                 calib_start_time = time.time()
                                 calib_base_ta = latest_gsr_ta
                                 calib_min_ta = latest_gsr_ta


                        if ovl: ovl.set_text(msg)
                            
                except Exception as e:
                    print(f"CALIB ERROR: {e}")
                    calib_mode = False # Abort
                
             else:
                 # Hide
                 if 'txt_calib_overlay' in ui_refs:
                     ui_refs['txt_calib_overlay'].set_text("")
                     ui_refs['txt_calib_overlay'].set_visible(False)

             # [NEW] specific update for Session Detail Panel
             if is_recording and recording_start_time:
                 elapsed = datetime.now() - recording_start_time
                 total_sec = int(elapsed.total_seconds())
                 hours, remainder = divmod(total_sec, 3600)
                 mins, secs = divmod(remainder, 60)
                 
                 if 'txt_sess_len' in ui_refs:
                     ui_refs['txt_sess_len'].set_text(f"Duration : {hours:02}:{mins:02}:{secs:02}")
                 # Ensure color is black
                 if 'txt_sess_len' in ui_refs: ui_refs['txt_sess_len'].set_color('black')
              
             else:
                 pass 
             


        elif current_view == 'settings':
             pass



    def on_close(event):
        global app_running
        app_running = False
        save_config() 
        print("Window Closed. Cleaning up...")

    fig.canvas.mpl_connect('close_event', on_close)

    # [FIX] Flush Last Log to UI
    if log_messages:
         try: log_msg(log_messages[-1].split("] ", 1)[-1])
         except: pass

    try:
        # [FIX] Custom Animation Loop (Force 50fps)
        timer = fig.canvas.new_timer(interval=20) 
        timer.add_callback(update)
        timer.start()

        # [FIX] Final Blt Logic appended to update()
        # This function runs at the END of update() to flush pixels
        def final_blit():
            # [FIX] Handle Settings View Simple Redraw
            if current_view == 'settings':
                 try:
                     fig.canvas.draw()
                     fig.canvas.flush_events()
                 except: pass
                 return

            if current_view != 'main': return
            
            # Draw Dynamic Artists
            if line: ax_graph.draw_artist(line)
            if 'txt_ta_set_line' in ui_refs: ax_graph.draw_artist(ui_refs['txt_ta_set_line'])
            
            # [FIX] Redraw Spines ON TOP of lines so they don't get covered
            for s in ax_graph.spines.values():
                ax_graph.draw_artist(s)
            
            # Draw Scores/Text
            try: ax_w_val.draw_artist(txt_win_val)
            except NameError: pass
            try: ax_scores.draw_artist(txt_ta_score)
            except NameError: pass
            try: ax_scores.draw_artist(val_txt)
            except NameError: pass
            
            # Draw System Line
            if 'system_line' in ui_refs: 
                 ax_info.draw_artist(ui_refs['system_line'])
            
            # Draw Counter
            try: ax_count_bg.draw_artist(txt_count_val)
            except NameError: pass
            
            # Draw Status
            if 'txt_gsr_status' in ui_refs: ax_status.draw_artist(ui_refs['txt_gsr_status'])
            try: ax_status.draw_artist(txt_hrm_status)
            except NameError: pass
            try: ax_status.draw_artist(txt_audio)
            except NameError: pass
            try: ax_status.draw_artist(rec_text)
            except NameError: pass
            
            # Draw Detail Panel Info
            if 'txt_sess_len' in ui_refs: ax_detail.draw_artist(ui_refs['txt_sess_len'])
            if 'txt_hr_val' in ui_refs: ax_detail.draw_artist(ui_refs['txt_hr_val'])
            if 'txt_hrv_val' in ui_refs: ax_detail.draw_artist(ui_refs['txt_hrv_val'])
            
            # Draw Pattern (Overlay on Graph)
            if 'txt_pattern' in ui_refs: ax_graph.draw_artist(ui_refs['txt_pattern'])

            # [NEW] Draw Overlays (Calibration / Motion)
            if 'txt_calib_overlay' in ui_refs and ui_refs['txt_calib_overlay'].get_visible():
                ax_overlay.draw_artist(ui_refs['txt_calib_overlay'])
            if 'txt_motion_overlay' in ui_refs and ui_refs['txt_motion_overlay'].get_visible():
                ax_overlay.draw_artist(ui_refs['txt_motion_overlay'])

            # [CRITICAL] Draw Buttons Manually so they overlay properly
            for b_key in ['btn_count', 'btn_reset', 'btn_ta_set_now', 'btn_rec', 'btn_to_settings', 'btn_back']:
                 if b_key in ui_refs and ui_refs[b_key].ax.get_visible():
                      b = ui_refs[b_key]
                      b.ax.draw_artist(b.ax.patch) # Background (Normal/Hover)
                      b.ax.draw_artist(b.label)    # Text
            
            # Blit All Regions
            if ax_graph: fig.canvas.blit(ax_graph.bbox)
            if ax_scores: fig.canvas.blit(ax_scores.bbox)
            if ax_count_bg: fig.canvas.blit(ax_count_bg.bbox)
            if ax_w_val: fig.canvas.blit(ax_w_val.bbox)
            if ax_status: fig.canvas.blit(ax_status.bbox)
            if ax_detail: fig.canvas.blit(ax_detail.bbox)
            if ax_info: fig.canvas.blit(ax_info.bbox) 
            # [CRITICAL] Blit Detached Buttons
            if 'btn_rec' in ui_refs: fig.canvas.blit(ui_refs['btn_rec'].ax.bbox)
            if 'btn_to_settings' in ui_refs and ui_refs['btn_to_settings'].ax.get_visible(): 
                fig.canvas.blit(ui_refs['btn_to_settings'].ax.bbox)
            if 'btn_back' in ui_refs and ui_refs['btn_back'].ax.get_visible(): 
                fig.canvas.blit(ui_refs['btn_back'].ax.bbox)
            
            fig.canvas.flush_events()
            
        # Monkey patch update to call final_blit
        original_update = update
        def update_wrapper(frame=0):
             original_update(frame)
             final_blit()
             
        # Re-bind timer
        timer.remove_callback(update)
        timer.add_callback(update_wrapper)
        plt.show()
    except Exception as e:
        import traceback
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        app_running = False
        print("Shutdown Initiated.")
        if f_gsr: f_gsr.close()

        # [NEW] Stop HRM and Close File
        if hrm_sensor:
             try: hrm_sensor.stop()
             except: pass
        if f_hrm: 
             try: f_hrm.close()
             except: pass
             
        if 'audio_handler' in globals() and audio_handler.audio_stream:
             audio_handler.audio_stream.close()
             
        # (t.join removed)
        print("Shutdown Complete.")
