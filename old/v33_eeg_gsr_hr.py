
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
from datetime import datetime, timedelta
from matplotlib.widgets import CheckButtons, RadioButtons, Button, Slider, TextBox  
from scipy.signal import butter, lfilter, lfilter_zi
from bleak import BleakClient, BleakScanner
import itertools
import os
import json 
import sounddevice as sd
import scipy.io.wavfile as wav
import tkinter as tk
from tkinter import ttk
import hid
from modules.bluetooth_handler import BluetoothHandler
from modules.audio_handler import AudioHandler
from modules.ant_driver import AntHrvSensor
from modules.hrv_state_analyzer import HRVStateAnalyzer
from modules.hrv_state_analyzer import HRVStateAnalyzer


# --- CONFIGURATION ---
FS = 250 # Hertz of headset datastream
# [REQ] Trend Smoothing Window (User variable)
# "Trend Window" = Moving Average Window
TREND_WINDOW_SEC = 2.0 # Default 1s smoothing
SMOOTHING_SAMPLES = int(FS * TREND_WINDOW_SEC)

BUFFER_SIZE = int(FS * 10) 
MAX_DISPLAY_SEC = 10 # Show 20 seconds of data on screen


WARMUP_SAMPLES = FS * 3
SCORE_SMOOTHING_WINDOW = int(FS * 2.0) 

# File Naming
CONFIG_FILE = "v33_config.json"

# --- GSR SETTINGS ---
VENDOR_ID = 0x1fc9
PRODUCT_ID = 0x0003
V_SOURCE = 6.371
R_REF = 83.0

# --- GLOBAL VARIABLES ---
latest_gsr_ta = 0.0
GSR_CENTER_VAL = 3.0
BASE_SENSITIVITY = 0.3 # [RENAMED] Was GSR_WINDOW_SIZE
booster_level = 0      # [NEW] 0=OFF, 1=LO, 2=MED, 3=HI
active_boost_level = 0 # [NEW] The level currently applied to graph zoom
CALIB_PIVOT_TA = 2.0
active_event_label = ""   # [NEW] The TA value where Sensitivity scaling is neutral (1.0x)
gsr_capture_queue = collections.deque(maxlen=100)

# --- HRM GLOBALS ---
latest_hr = 0
latest_hrv = 0.0
hrm_status = "Init"
hrm_sensor = None
hrv_analyzer = None # [NEW]
latest_hrm_state = None # [NEW]
txt_hr_val = None
txt_hrv_val = None

# [NEW] TA Blowdown Tracker
blowdown_peak_ta = 0.0
blowdown_triggered = False
blowdown_peak_time = 0.0 # [NEW] Track time of peak
blowdown_events = []     # [NEW] List of (peak_time, trigger_time) tuples
blowdown_path_len = 0.0  # [NEW] Accumulate total movement (efficiency check)
blowdown_prev_ta = 0.0   # [NEW] Track previous frame for delta

# [NEW] Global Motion Detector (Velocity Filter)
motion_lock_expiry = 0.0
motion_prev_ta = 0.0
motion_prev_time = 0.0
motion_lock_active = False # [NEW] State tracker for logging
session_start_time = time.time() # [NEW] Track app start for Warmup

def get_effective_window():
    # [REQ] Use active_boost_level (applied on Reset) not target booster_level
    if active_boost_level == 0: return BASE_SENSITIVITY
    
    # Logic from v20
    mult = [1.0, 0.6, 1.0, 1.4][active_boost_level]
    safe_ta = max(1.0, GSR_CENTER_VAL)
    try:
        # [FIX] Refactored for clarity:
        # Ratio = TA / CALIB_PIVOT. (e.g. 2.75 / 2.75 = 1.0)
        # We want Higher TA -> Smaller Window (Higher Sens).
        # So we raise to NEGATIVE mult. (Ratio ^ -mult).
        # If Ratio = 1.0 (Reset), Factor = 1.0 -> No Change.
        pivot = max(0.1, CALIB_PIVOT_TA) # Safety
        return BASE_SENSITIVITY * math.pow((safe_ta / pivot), -mult)
    except:
        return BASE_SENSITIVITY




# --- STATE ---
eeg_buffer = collections.deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
# [REQ] 60Hz * 10s = 600 samples (60Hz Plotting)
HISTORY_LEN = 600
bands_history = {k: collections.deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN) for k in
                 ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'GSR']}
                 
# [REQ] Smoothing Buffers for Trend Calculation
smoothing_buffers = {k: collections.deque(maxlen=10000) for k in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']}
smoothing_sums = {k: 0.0 for k in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']}

log_messages = collections.deque(["System Ready"], maxlen=10)
is_connected = False
current_state = "EEG: DISCONNECTED" 
current_theme = "Light" 
headset_on_head = False
app_running = True
counting_active = False # [NEW] TA Counter State
ta_accum = 0.0          # [NEW] TA Counter Total
is_recording = False
last_packet_time = 0
samples_since_contact = 0
device_battery_level = "--"
current_client = None
timestamp_queue = None
REAL_SERIAL_STR = None 
DEVICE_ADDRESS = None  
ui_update_queue = queue.Queue()
command_queue = queue.Queue()
raw_eeg_queue = queue.Queue() # [NEW] High-Res Sample Queue
ADVERTISED_NAME = "Unknown"

# Biofeedback
current_calm_score = 0
calm_history = collections.deque(maxlen=SCORE_SMOOTHING_WINDOW)

# Logic Settings
event_detected = False
last_on_signal_time = 0
total_samples_recorded = 0
latest_eeg_bands = {k: 0.0 for k in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']}

# Control Defaults
FFT_WINDOW_SEC = 1.0  
baseline_window_sec = 2.0
coincidence_window = 0.5
global_percent = 20
triggers_enabled = False

# CSV Handles
f_gsr = None
writer_gsr = None
f_trend = None
writer_trend = None
recording_start_time = None



# --- AUDIO RECORDING STATE ---
audio_filename = None

# --- DSP CACHE ---
dsp_cache = {
    'len': 0,        
    'window': None, 
    'freq': None,    
    'idx_d': None,   
    'idx_t': None,   
    'idx_a': None,   
    'idx_b': None,   
    'idx_g': None    
}

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
        # [NEW] Internal Buffer for High-Res Processing
        self.eeg_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.samples_to_process = 0.0 # [REQ] Pacing Accumulator

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
                    
                    # [NEW] Log Here to capture every sample (60Hz)
                    if is_recording:
                         try:
                             ts_now = datetime.now().strftime('%H:%M:%S.%f')
                             
                             # Log GSR
                             if writer_gsr:
                                  win = get_effective_window()
                                  global CALIB_PIVOT_TA, active_boost_level, active_event_label
                                  note = active_event_label
                                  if note: active_event_label = ""
                                  writer_gsr.writerow([ts_now, f"{self.current_ta:.5f}", f"{GSR_CENTER_VAL:.3f}", f"{1.0/win:.3f}", f"{win:.3f}", 0, f"{CALIB_PIVOT_TA:.3f}", active_boost_level, note])
                             
                             # Log Trend (Sampled at GSR rate)
                             if writer_trend:
                                  t_row = [ts_now, current_calm_score]
                                  has_data = False
                                  for k in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                                      if len(bands_history[k]) > 0:
                                          val = latest_eeg_bands[k]
                                          t_row.append(f"{val:.3f}")
                                          has_data = True
                                      else:
                                          t_row.append("0.000")
                                  
                                          t_row.append("0.000")
                                  
                                  # [REMOVED] HRM Data from EEG_Trend.csv
                                  # t_row.append(str(latest_hr))
                                  # t_row.append(f"{latest_hrv:.1f}")

                                  if has_data: 
                                       writer_trend.writerow(t_row)
                                       
                         except Exception as e:
                             pass

                    # [NEW] GRAPH HISTORY UPDATE (Master Clock = 60Hz)
                    try:
                         # 1. GSR Value
                         # [FIX] Store RAW TA for dynamic scaling in main loop
                         bands_history['GSR'].append(self.current_ta)
                         # [FIX] Performance Pruning: Limit history size
                         while len(bands_history['GSR']) > HISTORY_LEN:
                              bands_history['GSR'].pop(0)
                         
                         # 2. EEG High-Res Processing (Consumer Logic)
                         req_samples = int(FS * FFT_WINDOW_SEC)
                         
                         # [REQ] Paced Consumption
                         self.samples_to_process += (250.0 / 60.0)
                         if self.samples_to_process > 25.0: self.samples_to_process = 25.0
                         
                         count_to_process = int(self.samples_to_process)
                         self.samples_to_process -= count_to_process
                         
                         processed_count = 0
                         while not raw_eeg_queue.empty() and processed_count < count_to_process:
                             val = raw_eeg_queue.get()
                             self.eeg_buffer.append(val)
                             
                             if len(self.eeg_buffer) >= req_samples:
                                 clean_data = list(itertools.islice(self.eeg_buffer, len(self.eeg_buffer) - req_samples, len(self.eeg_buffer)))
                                 bands = calculate_relative_bands(clean_data)
                                 
                                 for i, k in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
                                     b_val = bands[i]
                                     target_len = int(FS * TREND_WINDOW_SEC) 
                                     if target_len < 1: target_len = 1
                                     
                                     buf = smoothing_buffers[k]
                                     if buf.maxlen != target_len:
                                         new_buf = collections.deque(list(buf), maxlen=target_len)
                                         smoothing_buffers[k] = new_buf
                                         buf = new_buf
                                         smoothing_sums[k] = sum(buf)
                                     
                                     if len(buf) == buf.maxlen:
                                         smoothing_sums[k] -= buf[0]
                                         
                                     buf.append(b_val)
                                     smoothing_sums[k] += b_val
                                     
                                     if len(buf) > 0: smoothed_val = smoothing_sums[k] / len(buf)
                                     else: smoothed_val = b_val
                                         
                                     latest_eeg_bands[k] = smoothed_val

                                 try:
                                     s_theta = latest_eeg_bands['Theta']
                                     s_alpha = latest_eeg_bands['Alpha']
                                     s_beta = latest_eeg_bands['Beta']
                                     denom = s_alpha + max(1e-6, s_beta)
                                     ratio = max(1e-6, s_theta) / (denom if denom > 0 else 1e-6)
                                     raw_c = 30 * math.log10(ratio) + 50
                                     current_calm_score = int(np.clip(raw_c, 0, 100))
                                 except: pass
                                 
                             processed_count += 1
                         
                         # 3. Plotting (60Hz Sample)
                         for k in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                             bands_history[k].append(latest_eeg_bands[k])
                    except: pass
                
                except Exception as loop_e:
                    print(f"[GSR] Loop Skip: {loop_e}")
                    time.sleep(0.005)

                time.sleep(0.015) 

        except Exception as e:
            print(f"[GSR] FATAL Error: {e}") 
            self.connected = False

    def stop(self):
        self.running = False


# --- FUNCTIONS ---
def calculate_relative_bands(clean_data):
    try:
        n = len(clean_data)
        if n == 0: return [0]*5
        if dsp_cache['len'] != n:
            dsp_cache['len'] = n
            dsp_cache['window'] = np.hanning(n)
            freqs = np.fft.rfftfreq(n, 1.0 / FS)
            dsp_cache['freq'] = freqs
            def get_idx(low, high):
                 return np.where((freqs >= low) & (freqs < high))
            dsp_cache['idx_d'] = get_idx(0.5, 4)
            dsp_cache['idx_t'] = get_idx(4, 8)
            dsp_cache['idx_a'] = get_idx(8, 13)
            dsp_cache['idx_b'] = get_idx(13, 30)
            dsp_cache['idx_g'] = get_idx(30, 50)

        windowed_data = clean_data * dsp_cache['window']
        fft_vals = np.abs(np.fft.rfft(windowed_data))
        def fast_pwr(indices, bw):
            if len(indices[0]) == 0: return 0
            return np.sum(fft_vals[indices]) / bw
        delta = fast_pwr(dsp_cache['idx_d'], 3.5)
        theta = fast_pwr(dsp_cache['idx_t'], 4.0)
        alpha = fast_pwr(dsp_cache['idx_a'], 5.0)
        beta  = fast_pwr(dsp_cache['idx_b'], 17.0)
        gamma = fast_pwr(dsp_cache['idx_g'], 20.0)
        total = delta + theta + alpha + beta + gamma
        if total == 0: return [0] * 5
        return [(delta / total) * 100, (theta / total) * 100, (alpha / total) * 100, (beta / total) * 100, (gamma / total) * 100]
    except: return [0] * 5

def log_msg(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    clean_msg = str(msg).strip()
    log_messages.append(f"[{timestamp}] {clean_msg}")
    print(clean_msg)




# --- CONFIG PERSISTENCE ---
def save_config():
    try:
        # [REQ] Save Graph Visibility from Checkbuttons
        # We need a way to access the current visibility state.
        # Since 'check' is inside the GUI setup, we might need a global ref or read from 'lines'.
        # Assuming 'lines' is global/accessible (it is in main view setup, but lines dict is populated in main).
        # We need to make sure 'lines' is accessible here. It is local to main block.
        # Wait, save_config is global. lines is defined in `if __name__ == "__main__":`.
        # We can't access `lines` directly if it's local.
        # However, `check_status` is not easily accessible unless we make it global.
        # Let's verify where `lines` and `check` are defined. They are inside `if __name__...`.
        # We should make `check` global or store state in a global dict `graph_visibility`.
        
        # ACTUALLY, we can save the Global Variables for GSR.
        vis_state = {}
        # We need to access the 'check' widget or 'lines' to know what is visible.
        # Solution: Use a global `active_graph_lines` list or similar.
        # Let's inspect line 700 area again.
        
        cfg = {
            'mic_name': audio_handler.current_mic_name if 'audio_handler' in globals() else "Default",
            'mic_gain': audio_handler.current_mic_gain if 'audio_handler' in globals() else 3.0,
            'mic_rate': audio_handler.current_mic_rate if 'audio_handler' in globals() else None,
            'bt_address': DEVICE_ADDRESS,
            'bt_name': ADVERTISED_NAME, 
            'gui_theme': current_theme,
            'gsr_center': GSR_CENTER_VAL,
            'gsr_base': BASE_SENSITIVITY, # [RENAMED]
            'booster_idx': booster_level  # [NEW]
        }
        
        # [NEW] Add Graph Visibility if global `ui_refs` has it or similar.
        # If we can't easily reach it without major refactor, we might skip or use a workaround.
        # Workaround: Add `current_graph_visibility` global and update it on click.
        if 'graph_visibility' in globals():
            cfg['graph_visibility'] = globals()['graph_visibility']
            
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

        # Restore Theme
        global current_theme
        current_theme = cfg.get('gui_theme', 'Light')
        
        # Restore GSR Settings
        global GSR_CENTER_VAL, BASE_SENSITIVITY, booster_level
        GSR_CENTER_VAL = float(cfg.get('gsr_center', 3.0))
        BASE_SENSITIVITY = float(cfg.get('gsr_base', 0.3)) 
        if 'gsr_base' not in cfg: BASE_SENSITIVITY = float(cfg.get('gsr_window', 0.3))
            
        booster_level = int(cfg.get('booster_idx', 0))
        
        # Restore Graph Visibility
        global initial_graph_visibility
        initial_graph_visibility = cfg.get('graph_visibility', None)
        
        # Restore Bluetooth
        global DEVICE_ADDRESS, ADVERTISED_NAME
        saved_addr = cfg.get('bt_address', None)
        saved_name = cfg.get('bt_name', "Unknown")
        if saved_addr:
             DEVICE_ADDRESS = saved_addr
             ADVERTISED_NAME = saved_name
             print(f"[Config] Restored Target Device: {ADVERTISED_NAME} ({DEVICE_ADDRESS})")
        
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
        # [NEW] Initialize Analyzer
        hrv_analyzer = HRVStateAnalyzer()
        print("[HRM] Sensor Started")
    except Exception as e:
        print(f"[HRM] Start Error: {e}")
    
    # Start Bluetooth Handler (Module)
    def update_status_cb(key, val):
        globals()[key] = val
        
    ble_handler = BluetoothHandler(raw_eeg_queue, command_queue, update_status_cb, log_msg)
    t = threading.Thread(target=lambda: asyncio.run(ble_handler.run()), daemon=True)

    t.start()
    
    # load_config() moved after AudioHandler init 
    pass
    
    # --- GUI ---
    fig = plt.figure(figsize=(15, 9)) # [REQ] Wider Window
    try: fig.canvas.manager.set_window_title("EEG Rolling Trend Viewer")
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
                 if 'txt_audio' in globals(): txt_audio.set_text(val)
        except: pass
        
    audio_handler = AudioHandler(log_msg, update_audio_ui)
    
    # Load Config AFTER AudioHandler exists
    load_config()
    
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
    # Adjusted width for Right Panel clearance
    # [0.05, 0.15, 0.78, 0.75] -> Shrink vertical.
    # New: Top same, Bottom higher. 
    # [0.05, 0.20, 0.78, 0.70] (Bottom 0.20 instead of 0.15)
    ax_graph = reg_ax([0.05, 0.20, 0.78, 0.70], main_view_axes)
    ax_graph.set_xlim(0, HISTORY_LEN)
    ax_graph.set_ylim(-5, 105)
    # [REQ] Grid dynamic visibility
    # ax_graph.grid(True, alpha=0.3) -> Init disabled
    ui_refs['ax_graph'] = ax_graph 
    
    ax_graph.set_title("Real-Time GSR & EEG Trends (Last ~10s)", fontsize=14, fontweight='bold')
    ax_graph.set_ylabel("Relative Power (%)")
    # ax_graph.set_xlabel("Time (samples)") # Hide for cleaner look
    ax_graph.set_xticks([])

    # Create lines
    lines = {}
    colors = {'Delta': 'blue', 'Theta': 'green', 'Alpha': 'orange', 'Beta': 'red', 'Gamma': 'purple', 'GSR': 'magenta'}
    
    # [REQ] Thicker GSR line
    linewidths = {'GSR': 2.5}
    
    for k, col in colors.items():
        lw = linewidths.get(k, 1.5)
        line, = ax_graph.plot([], [], lw=lw, color=col, label=k)
        lines[k] = line

    # [NEW] Blowdown Marker Line (Cyan Dashed)
    # This single line artist will draw dis-continuous segments using NaN
    line_blowdown, = ax_graph.plot([], [], color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    ui_refs['line_blowdown'] = line_blowdown

    # [NEW] Set Point Dashed Line

    # [NEW] Set Point Dashed Line
    # Fixed at 62.5% because our dynamic scale always places GSR_CENTER_VAL at that visual position.
    # [REQ] Deep Orange and Label
    ax_graph.axhline(y=62.5, color='#CC5500', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    txt_ta_set_line = ax_graph.text(0, 63.0, f"TA SET: {GSR_CENTER_VAL:.2f}", color='#CC5500', fontsize=8, fontweight='bold', ha='left')
    ui_refs['txt_ta_set_line'] = txt_ta_set_line
    
    # [NEW] Calibration Overlay Text (Figure Level for Visibility)
    txt_calib_overlay = fig.text(0.5, 0.5, "", ha='center', va='center', fontsize=24, fontweight='bold', color='red', zorder=100)
    # [NEW] Calibration Overlay Text (Figure Level for Visibility)
    txt_calib_overlay = fig.text(0.5, 0.5, "", ha='center', va='center', fontsize=24, fontweight='bold', color='red', zorder=100)
    ui_refs['txt_calib_overlay'] = txt_calib_overlay # Store ref
    
    # [NEW] Motion Overlay Text (Top Center)
    txt_motion_overlay = fig.text(0.5, 0.75, "", ha='center', va='center', fontsize=20, fontweight='bold', color='red', zorder=100)
    ui_refs['txt_motion_overlay'] = txt_motion_overlay
    
    # === CHECKBUTTON LEGEND (Right Side) ===
    # Shifted Right: [0.85, ...]
    ax_check = reg_ax([0.85, 0.65, 0.10, 0.25], main_view_axes)

    ax_check.set_facecolor('white')
    
    keys = list(colors.keys())
    # [REQ] Colored Circle Labels
    # We use a mapping to handle the key lookup in callback
    labels = [f"● {k}" for k in keys]
    label_map = {l: k for l, k in zip(labels, keys)}
    
    # [REQ] Restore Visibility from Config
    if 'initial_graph_visibility' in globals() and initial_graph_visibility:
        visibility = [initial_graph_visibility.get(k, True) for k in keys]
    else:
        visibility = [True] * len(keys)
        
    check = CheckButtons(ax_check, labels, visibility)

    # Sync Lines with Visibility immediately
    for i, k in enumerate(keys):
        lines[k].set_visible(visibility[i])

    # Global for Save
    global graph_visibility
    graph_visibility = {k: v for k, v in zip(keys, visibility)}

    def func(label):
        real_key = label_map[label]
        line = lines[real_key]
        is_vis = not line.get_visible()
        line.set_visible(is_vis)
        plt.draw()
        # Update global for save
        graph_visibility[real_key] = is_vis
        
    check.on_clicked(func)

    # Style the checkbuttons
    try:
        # [REQ] Color the Text Labels explicitly
        for i, lbl in enumerate(check.labels):
            k = keys[i]
            col = colors[k]
            lbl.set_color(col)
            lbl.set_fontweight('bold')
            # Ensure high visibility
            lbl.set_fontsize(11)

         # Keep default gray boxes but maybe slightly cleaner
        for r in check.rectangles:
            r.set_facecolor('#f0f0f0') 
            r.set_edgecolor('gray')
            
    except AttributeError: pass
            


    # === GSR CONTROLS CONTAINER ===
    # A single bordered panel to contain Scale, Sens, Boost, Calibrate
    # [REQ] "Balance the positioning"
    # Container Rect: X=0.835, Width=0.13, Top=0.58, Bottom=0.26
    # Container Rect: X=0.835, Width=0.13
    # [MOVED DOWN] To fit below Checkboxes (0.65). 
    # Target Bottom: 0.30. (Height 0.33 -> Top 0.63)
    r_ctrl = [0.835, 0.30, 0.13, 0.33] # H = 0.33
    ax_ctrl_bg = reg_ax(r_ctrl, main_view_axes)
    ax_ctrl_bg.set_facecolor('#f9f9f9')
    ax_ctrl_bg.set_xticks([]); ax_ctrl_bg.set_yticks([])
    # Border
    rect_ctrl_border = plt.Rectangle((0,0), 1, 1, transform=ax_ctrl_bg.transAxes, fill=False, ec='#aaaaaa', lw=2, clip_on=False)
    ax_ctrl_bg.add_patch(rect_ctrl_border)
    
    # --- 1. Title: GSR Scale ---
    # Relative to main axes to align easily
    # Relative to main axes to align easily
    # Relative to main axes to align easily
    # [MOVED DOWN] 0.69 -> 0.58
    ax_scale_lbl = reg_ax([0.835, 0.58, 0.13, 0.04], main_view_axes)
    ax_scale_lbl.set_axis_off()
    ax_scale_lbl.text(0.5, 0.5, "GSR Scale", ha='center', fontweight='bold', fontsize=12)
    
    # --- 2. Sensitivity ---
    # Label
    # Label
    # Label
    # [MOVED DOWN] 0.65 -> 0.54
    ax_win_lbl = reg_ax([0.835, 0.54, 0.13, 0.03], main_view_axes)
    ax_win_lbl.set_axis_off()
    ax_win_lbl.text(0.5, 0.5, "Sensitivity", ha='center', va='center', fontsize=10, fontweight='bold', color='#444')
    
    # Stepper [ - ] [ Val ] [ + ]
    # Centered in 0.835 + 0.13 = range 0.835 to 0.965. Center ~ 0.90
    # Centered in 0.835 + 0.13 = range 0.835 to 0.965. Center ~ 0.90
    y_sens = 0.50 # [MOVED DOWN] Was 0.61
    ax_w_down = reg_ax([0.85, y_sens, 0.03, 0.03], main_view_axes)
    ax_w_val  = reg_ax([0.88, y_sens, 0.04, 0.03], main_view_axes)
    ax_w_up   = reg_ax([0.92, y_sens, 0.03, 0.03], main_view_axes)
    
    ax_w_val.set_axis_off()
    
    def get_display_sens():
        w = get_effective_window()
        if w <= 0.001: return 99.9
        return 1.0 / w

    # Initial display
    txt_win_val = ax_w_val.text(0.5, 0.5, f"{get_display_sens():.2f}", ha='center', va='center', fontsize=10, fontweight='bold')
    
    btn_win_down = Button(ax_w_down, "-", color='lightgray', hovercolor='gray')
    btn_win_up   = Button(ax_w_up, "+", color='lightgray', hovercolor='gray')
    
    # --- 3. Auto-Boost ---
    # Title
    # Title
    # Title
    # [MOVED DOWN] 0.56 -> 0.45
    ax_boost_lbl = reg_ax([0.835, 0.45, 0.13, 0.02], main_view_axes)
    ax_boost_lbl.set_axis_off()
    ax_boost_lbl.text(0.5, 0.5, "Auto-Boost", ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    # Buttons [OFF] [L] [M] [H]
    # Spread evenly: 4 buttons in 0.13 width.
    # ~0.03 per button gap?
    # Spread evenly: 4 buttons in 0.13 width.
    # ~0.03 per button gap?
    y_b = 0.41 # [MOVED DOWN] Was 0.52
    # OFF: 0.845, L: 0.88, M: 0.905, H: 0.93
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
    
    # [MOVED] SET Button moved to central TA Box
    
    def update_boost_ui():
        for i, btn in enumerate(boost_btns):
            if i == booster_level:
                btn.color = 'lime'
                try: btn.label.set_color('black')
                except: pass
            else:
                btn.color = 'lightgray'
                try: btn.label.set_color('black')
                except: pass
            # [FIX] Force immediate color update
            try: btn.ax.set_facecolor(btn.color)
            except: pass
            
        # Trigger redraw 
        try: fig.canvas.draw_idle()
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
             
        plt.draw()

    btn_b_off.on_clicked(lambda e: set_boost(0))
    btn_b_lo.on_clicked(lambda e: set_boost(1))
    btn_b_med.on_clicked(lambda e: set_boost(2))
    btn_b_hi.on_clicked(lambda e: set_boost(3))
    
    # --- 4. Calibrate ---
    # Centered at bottom of panel
    # --- 4. Calibrate ---
    # Centered at bottom of panel
    # Centered at bottom of panel
    # [MOVED DOWN] 0.43 -> 0.32
    ax_calib = reg_ax([0.85, 0.32, 0.10, 0.04], main_view_axes)
    btn_calib = Button(ax_calib, "Calibrate", color='lightblue', hovercolor='cyan')
    
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
        
        calib_mode = True
        calib_phase = 0
        calib_step = 1
        calib_start_time = time.time()
        calib_base_ta = latest_gsr_ta
        calib_min_ta = latest_gsr_ta
        calib_vals = [] # Store drops
        
        log_msg(f"Calibration Started. Saving Boost Lvl: {saved_boost_level}")
        global active_event_label
        active_event_label = "CALIB_START"
        update_gsr_center(latest_gsr_ta)
        
    btn_calib.on_clicked(start_calibration)
    
    # --- 5. BIO-GRID (Start at Bottom) ---
    # Container at Bottom Right: [0.835, 0.05, 0.13, 0.20] (~Square visual)
    # [MOVED UP] 0.11 -> 0.22 (Up by 0.11)
    # The previous shift of Right Panel elements was +0.15 (0.26->0.41).
    # So the gap is 0.41 - 0.26 = 0.15.
    # Container at Bottom Right: [0.835, 0.05, 0.13, 0.20] (~Square visual)
    # [MOVED DOWN] 0.22 -> 0.14
    # Fits below Controls (0.30). Top at 0.27.
    
    r_bio = [0.835, 0.125, 0.13, 0.17] 
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
    # State: "Flow"
    # Trend: "Reset"
    # Text Below
    # State: "Flow"
    # Trend: "Reset"
    # [MOVED DOWN] 0.13 -> 0.05
    ax_grid_txt = reg_ax([0.835, 0.04, 0.13, 0.08], main_view_axes)
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
        
        delta = step * direction
        new_win = BASE_SENSITIVITY - delta
        BASE_SENSITIVITY = max(0.05, min(50.0, rounded_step(new_win)))
        txt_win_val.set_text(f"{get_display_sens():.2f}")
        plt.draw()
        
    def rounded_step(val): return round(val * 1000) / 1000.0 # Round to nearest 0.001

    # [FIX] Inverted Buttons: Up(+) adds Sensitivity (Reduces Window)
    btn_win_down.on_clicked(lambda e: change_sensitivity(-1)) # Reduce Sens = Add to Window
    btn_win_up.on_clicked(lambda e: change_sensitivity(1))    # Increase Sens = Subtract Window

    # [FIX] Moved TA SET to central box
    val_txt = None # Placeholder, defined below

    # [FIX] Added last_calib_ratio for debug
    last_calib_ratio = 0.0

    def update_gsr_center(val):
        global GSR_CENTER_VAL, ta_accum, calib_mode # [FIX] Added ta_accum, calib_mode
        global active_boost_level, booster_level    # [FIX] Added boost globals
        global motion_lock_expiry # [NEW]
        
        # [NEW] TA Counter Logic (Count Drops)
        if counting_active:
             # [REQ] Global Motion Block
             # If Motion Lock is active, do NOT count drops.
             if time.time() > motion_lock_expiry:
                 diff = GSR_CENTER_VAL - val
                 if diff > 0 and not calib_mode: # [REQ] Only count drops if not calibrating
                     ta_accum += diff

        # [FIX] Apply Auto-Boost Logic on Reset/Re-Center
        # Only if NOT calibrating (Calibration forces manual zoom)
        if not calib_mode:
            # [REQ] Calibration Pivot Logic
            # When switching from Manual (0) to Auto (>0), we set the Pivot.
            # This ensures that RIGHT NOW (at this TA), the scaling factor is 1.0 (No Jump).
            if active_boost_level == 0 and booster_level > 0:
                 global CALIB_PIVOT_TA
                 CALIB_PIVOT_TA = max(0.1, val)
                 log_msg(f"Boost Engage: Pivot Set to {CALIB_PIVOT_TA:.2f}")

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
                active_event_label = "USER_TA_RESET"
                active_boost_level = booster_level
                update_gsr_center(latest_gsr_ta)

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # === TA & SCORES PANELS (Clean Re-implementation) ===
    # Moved UP to avoid overlapping System Info (Y=0.04)
    # Graph Bottom is now 0.20, so we have 0.04 to 0.20 to work with (Height 0.16)
    
    # 1. TA Counter Panel (Right Side of Left Block)
    # [0.22, 0.05, 0.18, 0.12] 
    r_count = [0.20, 0.05, 0.20, 0.12]
    ax_count_bg = reg_ax(r_count, main_view_axes)
    ax_count_bg.set_xticks([]); ax_count_bg.set_yticks([])
    
    # Background Patch (Dynamic Color)
    # [FIX] Explicit Z-order on patch
    bg_count_rect = plt.Rectangle((0,0), 1, 1, transform=ax_count_bg.transAxes, color='#ffcccc', ec='black', lw=2, clip_on=False)
    bg_count_rect.set_zorder(0) # Patch at bottom of Axes
    ax_count_bg.add_patch(bg_count_rect)
    ui_refs['count_bg_rect'] = bg_count_rect
    
    # Text
    txt_count_val = ax_count_bg.text(0.5, 0.70, "TA Counter: 0.00", ha='center', va='center', fontsize=16, fontweight='bold', color='#550000')
    txt_count_val.set_zorder(10)
    
    # Buttons (Axes inside the panel area relative to Figure)
    # Panel: X 0.22-0.40. Y 0.05-0.17.
    # Btn Y: 0.06. Height 0.04.
    # Start: X=0.23, W=0.07
    # Reset: X=0.31, W=0.07
    ax_btn_start = reg_ax([0.23, 0.06, 0.07, 0.04], main_view_axes)
    ax_btn_reset = reg_ax([0.31, 0.06, 0.07, 0.04], main_view_axes)
    
    # Ensure Z-Order (Axes Level)
    ax_count_bg.set_zorder(1)
    ax_btn_start.set_zorder(100)
    ax_btn_reset.set_zorder(100)
    
    ui_refs['btn_count'] = Button(ax_btn_start, 'Start', color='lightgray', hovercolor='gray')
    ui_refs['btn_reset'] = Button(ax_btn_reset, 'Reset', color='#ffcccc', hovercolor='red')
    
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
        b.color = 'lightgreen' if counting_active else 'lightgray'
        b.hovercolor = 'green' if counting_active else 'gray'
        
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

        fig.canvas.draw_idle()

    def reset_count(e):
        if counting_active: return
        global ta_accum
        ta_accum = 0.0
        txt_count_val.set_text(f"TA Counter: {ta_accum:.2f}")
        fig.canvas.draw_idle()

    ui_refs['btn_count'].on_clicked(toggle_count)
    ui_refs['btn_reset'].on_clicked(reset_count)


    # 2. Scores Panel (Center)
    # [0.42, 0.05, 0.18, 0.12]
    r_score = [0.42, 0.05, 0.18, 0.12]
    ax_scores = reg_ax(r_score, main_view_axes)
    ax_scores.set_xticks([]); ax_scores.set_yticks([])
    ax_scores.set_facecolor('#e8e8e8')
    # Border
    rect_score_border = plt.Rectangle((0,0), 1, 1, transform=ax_scores.transAxes, fill=False, ec='black', lw=2, clip_on=False)
    ax_scores.add_patch(rect_score_border)

    txt_ta_score = ax_scores.text(0.5, 0.75, "INST TA: --", ha='center', va='center', fontsize=16, fontweight='bold', color='black')
    val_txt = ax_scores.text(0.5, 0.50, f"TA SET: {GSR_CENTER_VAL:.2f}", ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')

    # SET Button
    # Panel X 0.42-0.60. Center 0.51.
    # Btn: X=0.48, Y=0.06, W=0.06
    ax_btn_set = reg_ax([0.48, 0.06, 0.06, 0.04], main_view_axes)
    btn_ta_set_now = Button(ax_btn_set, "SET", color='lightblue', hovercolor='cyan')
    
    def force_set_center(e=None): 
        if latest_gsr_ta > 0.01:
             update_gsr_center(float(f"{latest_gsr_ta:.2f}"))
    btn_ta_set_now.on_clicked(force_set_center)
    
    # === STATUS BAR (Restored) ===
    ax_status = reg_ax([0.05, 0.94, 0.90, 0.04], main_view_axes)
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    ax_status.set_facecolor('#333')
    
    txt_gsr_status = ax_status.text(0.02, 0.5, "GSR: ●", color='lightgray', fontsize=11, fontweight='bold', va='center')
    txt_hrm_status = ax_status.text(0.15, 0.5, "HRM: ●", color='lightgray', fontsize=11, fontweight='bold', va='center')
    txt_conn = ax_status.text(0.28, 0.5, "EEG: ●", color='lightgray', fontsize=11, fontweight='bold', va='center')
    txt_wear = ax_status.text(0.41, 0.5, "FIT: --", color='lightgray', fontsize=11, fontweight='bold', va='center') # Shortened to FIT
    # [FIX] Use Noto Emoji (Monochrome) as Color Emoji often fails in Matplotlib
    txt_batt = ax_status.text(0.54, 0.5, "BATT: --", color='lightgray', fontsize=11, fontweight='bold', va='center', family=['Noto Emoji', 'sans-serif'])
    txt_audio = ax_status.text(0.67, 0.5, "AUDIO: --", color='lightgray', fontsize=11, fontweight='bold', va='center')
    
    # Rec moved left for balance (0.96 -> 0.92)
    rec_text = ax_status.text(0.92, 0.5, "● REC", color='red', fontsize=11, fontweight='bold', va='center', visible=False)
    
    ui_refs['txt_gsr_status'] = txt_gsr_status

    # Record Button (Moved Left)
    r_rc = [0.05, 0.12, 0.12, 0.05]
    ax_rec = reg_ax(r_rc, main_view_axes)
    ui_refs['btn_rec'] = Button(ax_rec, 'Record', color='lightgreen')
    
    import tkinter as tk

    # [NEW] Globals for Auto-Start Sequence
    pending_rec = False
    pending_notes = ""
    session_start_ta = 0.0

    def start_actual_recording():
        global is_recording, f_gsr, writer_gsr, f_trend, writer_trend, recording_start_time
        global f_hrm, writer_hrm # [NEW] HRM File
        global notes_filename, audio_filename # [FIX] Restored audio_filename
        # [FIX] Audio Globals Removed (Using AudioHandler)
        global pending_rec, pending_notes, session_start_ta, counting_active, ta_accum # [FIX] Globals
        
        try:
             # Capture Start Stats
             session_start_ta = latest_gsr_ta
             
             # Reset TA Counter
            #  if counting_active: toggle_count(None) # Stop first? No, actually we want it RUNNING.
            #  User said: "then after that, TA Counter is initiated (if it was already started 0- then reset and restart)"
             reset_count(None) # Reset to 0
             # if not counting_active: toggle_count(None) # [REQ] Delay counting until Calib finishes
             
             # Create Files
             DATA_DIR = "Session_Data"
             ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
             SESSION_DIR = os.path.join(DATA_DIR, f"Session_{ts_str}")
             os.makedirs(SESSION_DIR, exist_ok=True)
             
             fname_gsr = os.path.join(SESSION_DIR, "GSR.csv")
             fname_trend = os.path.join(SESSION_DIR, "EEG_Trend.csv")
             fname_hrm = os.path.join(SESSION_DIR, "HRM.csv") # [NEW]
             audio_filename = os.path.join(SESSION_DIR, "audio.wav")
             notes_filename = os.path.join(SESSION_DIR, "notes.txt")
             
             with open(notes_filename, "w") as f:
                 f.write(f"Session Notes - {ts_str}\n")
                 f.write("-" * 30 + "\n")
                 f.write(pending_notes if pending_notes else "No notes provided.")
             
             # Initialize GSR CSV
             f_gsr = open(fname_gsr, 'w', newline='')
             writer_gsr = csv.writer(f_gsr)
             writer_gsr.writerow(["Timestamp", "TA", "TA SET", "Sensitivity", "Window_Size", "Motion", "Pivot", "Boost", "Notes"])
             
             # Initialize Trend CSV (Reverted to standard EEG cols)
             f_trend = open(fname_trend, 'w', newline='')
             writer_trend = csv.writer(f_trend)
             writer_trend.writerow(["Timestamp", "Calm_Score", "Delta", "Theta", "Alpha", "Beta", "Gamma"])

             # [NEW] Initialize HRM CSV
             f_hrm = open(fname_hrm, 'w', newline='')
             writer_hrm = csv.writer(f_hrm)
             # Columns: Timestamp, HR_BPM, RMSSD_MS, Raw_RR_MS, State, Trend, Status, Raw_Packet_Hex, Z_HR, Z_HRV, Quadrant
             writer_hrm.writerow(["Timestamp", "HR_BPM", "RMSSD_MS", "Raw_RR_MS", "State", "Trend", "Status", "Raw_Packet_Hex", "Z_HR", "Z_HRV", "Quadrant"])

             audio_handler.audio_buffer = []
             is_recording = True 
             audio_handler.is_recording = True
             audio_handler.sync_audio_stream(current_view)
             
             recording_start_time = datetime.now()
             
             ui_refs['btn_rec'].label.set_text("Stop")
             ui_refs['btn_rec'].color = 'salmon'
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
        global is_recording, f_gsr, writer_gsr, f_trend, writer_trend, recording_start_time
        global f_hrm, writer_hrm # [NEW]
        global notes_filename, audio_filename # [FIX] Restored audio_filename
        # [FIX] Audio Globals Removed (Using AudioHandler)
        global pending_rec, pending_notes, calib_mode, calib_step, calib_phase, calib_start_time, calib_base_ta, calib_min_ta, counting_active # [FIX] Added counting_active
        
        if not is_recording:
             if audio_handler.selected_device_idx is None:
                 log_msg("Err: No Mic Selected!")
                 return
                 
             root = tk.Tk(); root.withdraw()
             note_data = {"text": None}
             
             dlg = tk.Toplevel(root)
             dlg.title("Session Notes")
             dlg.geometry("500x400")
             
             tk.Label(dlg, text="Enter Session Details:", font=("Arial", 10, "bold")).pack(pady=5)
             template_text = "Client Name: \n\nProcess Run: \n\nOther Notes: \n\n"
             
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
             ui_refs['btn_rec'].color = 'lightgreen'
             plt.draw() 
             
             if f_gsr: f_gsr.close()
             if f_trend: f_trend.close()
             if f_hrm: f_hrm.close() # [NEW]
             
             if audio_handler.audio_buffer:
                 audio_handler.save_audio(audio_filename)
             audio_handler.sync_audio_stream(current_view) 
             
             # ui_refs['btn_rec'] updated above
             
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
    
    # 3. Session Details Panel (Right of Scores)
    # Scores Panel ends at 0.60 (0.42 + 0.18)
    # Available width: 0.62 to 0.85
    r_detail = [0.62, 0.05, 0.20, 0.12]
    ax_detail = reg_ax(r_detail, main_view_axes)
    ax_detail.set_xticks([]); ax_detail.set_yticks([])
    ax_detail.set_facecolor('#f9f9f9')
    rect_det_border = plt.Rectangle((0,0), 1, 1, transform=ax_detail.transAxes, fill=False, ec='#aaaaaa', lw=2, clip_on=False)
    ax_detail.add_patch(rect_det_border)
    
    # Title
    ax_detail.text(0.5, 0.85, "Session Detail", ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    
    # Text Fields
    # "Date: --"
    # "Time: --"
    # "Length: --"
    txt_sess_date = ax_detail.text(0.05, 0.60, "Date: --", ha='left', va='center', fontsize=9, color='#333')
    txt_sess_time = ax_detail.text(0.05, 0.40, "Time: --", ha='left', va='center', fontsize=9, color='#333')
    txt_sess_len  = ax_detail.text(0.05, 0.15, "Duration : 00:00:00", ha='left', va='center', fontsize=10, fontweight='bold', color='black')
    
    ui_refs['txt_sess_date'] = txt_sess_date
    ui_refs['txt_sess_time'] = txt_sess_time
    ui_refs['txt_sess_len'] = txt_sess_len

    # [NEW] Vitals Display (HR/HRV)
    # Using spare vertical space in Detail panel
    txt_hr_val = ax_detail.text(0.55, 0.60, "HR: --", ha='left', va='center', fontsize=10, fontweight='bold', color='red')
    txt_hrv_val = ax_detail.text(0.55, 0.40, "HRV: --", ha='left', va='center', fontsize=10, fontweight='bold', color='blue')
    ui_refs['txt_hr_val'] = txt_hr_val
    ui_refs['txt_hrv_val'] = txt_hrv_val

    r_ts = [0.05, 0.06, 0.12, 0.05]
    ax_to_set = reg_ax(r_ts, main_view_axes)
    ui_refs['btn_to_settings'] = Button(ax_to_set, "Settings >", color='lightgray')

    # --- SETTINGS PAGE ELEMENTS ---
    rect_gui  = [0.05, 0.28, 0.28, 0.25] 
    rect_audio = [0.35, 0.28, 0.28, 0.25] 
    rect_conf = [0.65, 0.28, 0.33, 0.25] 
    rect_log = [0.65, 0.05, 0.33, 0.20] 
    
    ax_gui_bg  = create_panel_ax(rect_gui, "GUI Settings") 
    ax_conf_bg = create_panel_ax(rect_conf, "Device Controls")
    ax_audio_bg = create_panel_ax(rect_audio, "Audio Input Control")
    
    # 0. GUI Settings (Theme)
    r_thm = [rect_gui[0] + 0.02, rect_gui[1] + 0.14, 0.10, 0.08]
    ax_thm = reg_ax(r_thm, settings_view_axes)
    ax_thm.set_facecolor('#e0e0e0'); ax_thm.axis('off')
    
    is_dark = (current_theme == 'Dark')
    # [FIX] Set explicit facecolor for axis to avoid transparency issues
    ax_thm.set_facecolor('#e0e0e0') 
    
    ui_refs['rad_theme'] = RadioButtons(ax_thm, ['Light', 'Dark'], active=(1 if is_dark else 0))
    # [FIX] Style RadioButtons immediately
    try:
        for lbl in ui_refs['rad_theme'].labels: 
            lbl.set_fontsize(10)
            lbl.set_fontweight('bold')
        for c in ui_refs['rad_theme'].circles:
            c.set_radius(0.12) # Slightly larger
    except: pass
    
    def apply_theme(theme_name):
        global current_theme
        current_theme = theme_name
        
        if theme_name == 'Dark':
            c_bg = '#1e1e1e'; c_fg = 'white'; c_panel = '#383838'; c_grid = '#555555'; c_ax = '#2b2b2b'
        else: 
            c_bg = '#f0f0f0'; c_fg = 'black'; c_panel = '#e0e0e0'; c_grid = 'gray'; c_ax = 'white'
            
        fig.patch.set_facecolor(c_bg)
        
        # Apply to Graph
        try:
             ax_graph.set_facecolor(c_ax)
             ax_graph.tick_params(colors=c_fg)
             ax_graph.xaxis.label.set_color(c_fg); ax_graph.yaxis.label.set_color(c_fg)
             ax_graph.title.set_color(c_fg)
             for spine in ax_graph.spines.values(): spine.set_color(c_fg)
             # Legend?
             legend = ax_graph.get_legend()
             if legend:
                 for text in legend.get_texts(): text.set_color('black') # Force black for legend text often better
        except: pass
        
        # Apply to Scores panel
        try:
            ax_scores.set_facecolor(c_panel)
            txt_ta_score.set_color('white' if theme_name == 'Dark' else 'black')
        except: pass
        
        # Apply to Status Bar - DISABLED to enforce Fixed Dark Theme
        # try:
        #     ax_status.set_facecolor('#333') # Always Dark
        #     # Text colors handled by update() and setup_ui() to be fixed LightGray
        # except: pass
        
        # Apply to Settings panels
        try:
            panels = [ax_gui_bg, ax_conf_bg, ax_audio_bg, ax_log_bg]
            for p in panels:
                p.set_facecolor(c_panel)
                for art in p.texts: art.set_color(c_fg)
        except: pass
        
        try:
            radios = [
                 (ax_thm, ui_refs['rad_theme']), 
            ]
            for a, r in radios:
                a.set_facecolor(c_panel)
                try:
                    for l in r.labels: l.set_color(c_fg)
                except: pass
        except: pass
        
        try: system_line.set_color(c_fg)
        except: pass
        
        try: log_text.set_color(c_fg)
        except: pass
        
        plt.draw()
        log_msg(f"Theme: {theme_name}")

    def on_theme_change(label):
        apply_theme(label)
    ui_refs['rad_theme'].on_clicked(on_theme_change)
    
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

    # 4. DEVICE CONTROLS
    left_x = rect_conf[0] + 0.02

    # Rename (Bottom)
    r_name = [rect_conf[0] + 0.02, rect_conf[1] + 0.005, 0.18, 0.04]
    ax_name = reg_ax(r_name, settings_view_axes)
    ui_refs['text_name'] = TextBox(ax_name, '', initial="Rename...", label_pad=0.02)
    
    r_save = [rect_conf[0] + 0.22, rect_conf[1] + 0.005, 0.08, 0.04]
    ax_save = reg_ax(r_save, settings_view_axes)
    ui_refs['btn_name_save'] = Button(ax_save, 'Set', color='lightgreen')
    
    ignore_ui_callbacks = False
    def submit_name(text=None):
        if ignore_ui_callbacks: return
        if text is None: text = ui_refs['text_name'].text
        if len(text) > 0 and text != "...":
             s_bytes = text.strip().encode('utf-8')
             if len(s_bytes) > 0:
                 pl = b'\x1a' + bytes([len(s_bytes)]) + s_bytes
                 # send_command(10, pl) -> Use command_queue
                 command_queue.put_nowait((10, pl))
                 log_msg(f"Device Rename queued: {text}")
    ui_refs['text_name'].on_submit(submit_name)
    ui_refs['btn_name_save'].on_clicked(lambda e: submit_name())

    r_bk = [0.05, 0.90, 0.10, 0.04]
    ax_back = reg_ax(r_bk, settings_view_axes)
    ui_refs['btn_back'] = Button(ax_back, "< Back", color='lightgray')

    ax_log_bg = create_panel_ax(rect_log, "System Log")
    r_lg = [rect_log[0] + 0.02, rect_log[1] + 0.02, rect_log[2] - 0.04, 0.16]
    ax_log = reg_ax(r_lg, settings_view_axes)
    ax_log.axis('off')
    log_text = ax_log.text(0, 1, "", va="top", fontsize=8, family='monospace')

    ax_info = plt.axes([0.00, 0.00, 1.00, 0.04])
    ax_info.set_axis_off()
    system_line = ax_info.text(0.5, 0.5, "Waiting for Info...", ha="center", family=['Arial', 'Noto Emoji', 'sans-serif'])
    
    def teleport_off(ax_list):
        for a in ax_list: a.set_visible(False); a.set_position([1.5, 1.5, 0.01, 0.01])

    def teleport_on(ax_list):
        for a in ax_list: a.set_visible(True); a.set_position(ax_positions[a])

    teleport_off(settings_view_axes)
    teleport_on(main_view_axes)

    def req_main(e): 
        global desired_view; desired_view = 'main'
        audio_handler.sync_audio_stream('main')
    def req_settings(e): 
        global desired_view; desired_view = 'settings'
        audio_handler.sync_audio_stream('settings')
    
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
        plt.draw()
        
    # [FIX] Removed duplicate check_auto_center definition to avoid conflict with inline logic
    
    def update(frame):
        global current_view, desired_view, current_state, event_detected, ignore_ui_callbacks
        global headset_on_head # [FIX] Ensure we read the updated global
        global BASE_SENSITIVITY, saved_boost_level, booster_level # [FIX] For Calibration
        global calib_mode, calib_phase, calib_step, calib_start_time, calib_step_start_time, calib_base_ta, calib_min_ta, calib_vals, last_calib_ratio # [FIX] Added last_calib_ratio
        global recording_start_time, is_recording, session_start_ta # [FIX] Stats Update
        global pending_rec # [FIX] For Calibration Auto-Start
        global latest_hr, latest_hrv, hrm_status # [NEW] HRM Globals
        global active_event_label
        
        global is_connected_prev # [NEW] Track prev state
        global prev_grid_state
        
        # [NEW] Auto-Center on First Valid Reading
        global first_run_center
        if 'first_run_center' not in globals(): first_run_center = False

        if not first_run_center and latest_gsr_ta > 0.1:
             update_gsr_center(latest_gsr_ta)
             first_run_center = True
        
        if is_connected:
            if time.time() - last_packet_time > 1.5: current_state = "EEG: DISCONNECTED"
            else: current_state = "EEG: STREAMING"
        else: current_state = "EEG: DISCONNECTED"
        
        # [REQ] Dynamic Grid Visibility
        # Show grid only if EEG is connected
        should_show_grid = is_connected
        if 'prev_grid_state' not in globals(): prev_grid_state = None
        
        if should_show_grid != prev_grid_state:
             prev_grid_state = should_show_grid
             try: 
                 if should_show_grid:
                     ax_graph.grid(True, alpha=0.3)
                 else:
                     ax_graph.grid(False)
                     
                 ax_graph.get_yaxis().set_visible(should_show_grid) # [REQ] Hide Scale if unconnected
             except: pass

        
        if is_recording: rec_text.set_alpha(1.0 if frame % 10 < 5 else 0.3)
        else: rec_text.set_alpha(0.0)
        
        if current_view != desired_view:
            if desired_view == 'settings': teleport_off(main_view_axes); teleport_on(settings_view_axes)
            elif desired_view == 'main': teleport_off(settings_view_axes); teleport_on(main_view_axes)
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
            if hrv_analyzer:
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
        
        # [MODIFIED] Construct System Line Parts
        sys_parts = []
        if is_connected:
            sys_parts.append(f"Name: {dev_static.get('name','?')} | SN: {dev_static.get('serial','?')} | HW: {dev_static.get('hw','?')} | FW: {dev_static.get('fw','?')} | Manuf: {dev_static.get('manuf','?')} | Batt: 🔋 {device_battery_level}")
        
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
                    
                    if writer_hrm:
                        # Timestamp, HR_BPM, RMSSD_MS, Raw_RR_MS, State, Trend, Status, Raw_Packet_Hex
                        # Use Analyzer Output for State and Trend
                        log_state = f"{s_state}:{s_intens}"
                        z_h_log = analyzer_res.get('z_hr', '0.000')
                        z_v_log = analyzer_res.get('z_hrv', '0.000')
                        quad_log = analyzer_res.get('quadrant', 0)
                        
                        writer_hrm.writerow([ts_hrm, latest_hr, int(latest_hrv), raw_rr, log_state, s_trend, h_stat, raw_hex, z_h_log, z_v_log, quad_log])
                except: pass

        else:
            txt_hrm_status.set_color('gray')
            
        txt_conn.set_text("EEG: ●")
        if "STREAMING" in current_state:
             txt_conn.set_color('#009900') # Darker Green
        else:
             txt_conn.set_color('red')

        if current_state == "EEG: STREAMING":
            txt_wear.set_text("HEADSET : ON HEAD" if headset_on_head else "HEADSET : OFF HEAD")
            txt_wear.set_color("#009900" if headset_on_head else "red") 
            
            # txt_calm_score.set_text(f"Calm: {current_calm_score}")
        else:
            # [FIX] Reset Headset status when not streaming
            txt_wear.set_text("HEADSET: --")
            txt_wear.set_color('lightgray')
            # txt_calm_score.set_text("Calm: --")
        
        # [FIX] Always update Battery Status (even if Disconnected) to ensure Color is correct
        txt_batt.set_text(f"BATT: {device_battery_level}")
        
        # Default to standard 'lightgray'
        batt_col = 'lightgray' 
        
        # Color Logic
        if device_battery_level != "--":
            try:
                b_val = int(device_battery_level.replace('%',''))
                if b_val > 60: batt_col = '#009900' # Dark Green
                elif b_val < 20: batt_col = 'red'
            except: pass
        
        txt_batt.set_color(batt_col)
        
        # [Moved] Update GSR UI
        # 1. Update Text
        txt_ta_score.set_text(f"INST TA: {latest_gsr_ta:.3f}")
        txt_count_val.set_text(f"TA Counter: {ta_accum:.2f}") # [NEW] Update Counter
        
        # [NEW] Update GSR Status Label (Light)
        if 'txt_gsr_status' in ui_refs:
             if gsr_thread and gsr_thread.connected:
                  ui_refs['txt_gsr_status'].set_text("GSR: ●")
                  ui_refs['txt_gsr_status'].set_color('#009900') # Darker Green
             else:
                  ui_refs['txt_gsr_status'].set_text("GSR: ●")
                  ui_refs['txt_gsr_status'].set_color('red')
        
        # [NEW] Effective Window for Display
        eff_win = get_effective_window()
        # [FIX] Display as Sensitivity (Inverted)
        disp_sens = 99.9 if eff_win <= 0.001 else (1.0 / eff_win)
        txt_win_val.set_text(f"{disp_sens:.2f}")

        # [NEW] Update TA SET Line Label
        if 'txt_ta_set_line' in ui_refs and ui_refs['txt_ta_set_line']:
             ui_refs['txt_ta_set_line'].set_text(f"TA SET: {GSR_CENTER_VAL:.2f}")

        # 3. Auto-Center Logic
        # [REQ] Only if Auto-Boost is active (OFF = Manual)
        if latest_gsr_ta > 0.01 and booster_level != 0: # Only if valid signal and Boost ON
            # Recalculate limits for check
            # [FIX] Use effective window
            # [REQ] Updated Split: 0.625 Bottom / 0.375 Top
            min_p = GSR_CENTER_VAL - (0.625 * eff_win)
            max_p = GSR_CENTER_VAL + (0.375 * eff_win)
            
            if latest_gsr_ta < min_p or latest_gsr_ta > max_p:
                # Auto-Center
                # [REQ] Apply pending Auto-Boost Zoom on Auto-Reset
                active_boost_level = booster_level
                update_gsr_center(latest_gsr_ta)
            

        should_update_graph = (current_state == "EEG: STREAMING") or (current_view == 'main')
        
        if should_update_graph and current_view == 'main':
             # UPDATE GRAPH
             # [OPT] Only update visible lines.
             # [OPT] Skip EEG processing if disconnected
             
             for k, line in lines.items():
                 if not line.get_visible(): continue
                 
                 # [OPT] Skip EEG bands if not connected
                 if k != 'GSR' and not is_connected: # Assuming global is_connected
                      if len(line.get_xdata()) > 0: line.set_data([], [])
                      continue
                 
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
                     # Using stride of 2
                     # Convert to numpy array immediately for speed
                     raw_ys = np.array(list(itertools.islice(data, 0, len(data), 2)))
                     
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

        # Update Audio Meter (INDEPENDENT of EEG State)
        # Update Audio Meter (INDEPENDENT of EEG State)
        if audio_handler.audio_stream and audio_handler.audio_stream.active:
             raw_lvl = audio_handler.audio_state['peak']
             lvl = min(1.0, raw_lvl) 
             
             bar_level[0].set_width(lvl)
             ui_refs['text_level'].set_text(f"{int(lvl*100)}%") 
             
             if lvl > 0.95: bar_level[0].set_color('red') 
             elif lvl > 0.6: bar_level[0].set_color('yellow')
             else: bar_level[0].set_color('lime')
        else:
             bar_level[0].set_width(0)
             ui_refs['text_level'].set_text("OFF")

        # [FIX] Declare bg_rect global if not already mapped via ui_refs or similar.
        # Since I defined bg_rect in setup_ui but didn't put it in ui_refs or make it global, I need to fix that.
        # But wait, 'bg_rect' is local to setup_ui in my previous edit. I need to fix scope. 
        # Actually simplest to put it in ui_refs.
        
        # [NEW] Add Auto-Center Check to end of loop
        if not calib_mode and current_view == 'main':
             # [REQ] TA Blowdown Detection (Sudden Drop >= 0.2)
             global blowdown_peak_ta, blowdown_triggered, blowdown_peak_time, blowdown_events # [FIX] Added events list
             global blowdown_path_len, blowdown_prev_ta # [NEW] Motion Tracking
             global motion_lock_expiry, motion_prev_ta, motion_prev_time # [NEW] Global Motion
             
             # [REQ] GLOBAL VELOCITY FILTER
             # Calculate Speed (TA/sec)
             now = time.time()
             if motion_prev_time == 0.0: 
                  motion_prev_time = now
                  motion_prev_ta = latest_gsr_ta
             
             dt = now - motion_prev_time
             if dt > 0.0:
                  velocity = abs(latest_gsr_ta - motion_prev_ta) / dt
                  # Threshold: > 2.0 TA/s is physically impossible (Motion)
                  # Threshold: > 2.0 TA/s is physically impossible (Motion)
                  # [REQ] Hardened: 1.5 TA/s Threshold, 2.0s Duration
                  if velocity > 1.5:
                       motion_lock_expiry = now + 2.0 # Lock for 2s
                       # [REQ] Log Once per engagement
                       global motion_lock_active
                       if not motion_lock_active:
                            active_event_label = "MOTION_LOCK_ENGAGED"
                            motion_lock_active = True
                       
                  # [NEW] Manage Motion Overlay & State
                  # Use ui_refs to update text
                  # Priority: Velocity Lock > Steadiness High
                  
                  overlay_text = ""
                  if now < motion_lock_expiry:
                       overlay_text = f"MOTION (Vel={velocity:.1f})" if 'velocity' in locals() else "MOTION LOCK"
                       if 'txt_motion_overlay' in ui_refs:
                           ui_refs['txt_motion_overlay'].set_text(overlay_text)
                  else:
                       # If Velocity is OK, check if we are UNSTEADY but trying to Reset?
                       # We only check steadiness if we are Out of Bounds.
                       # But maybe we show "Unsteady" if range is high regardless?
                       # Let's check history just for the UI
                       if 'txt_motion_overlay' in ui_refs:
                           history = list(itertools.islice(bands_history['GSR'], max(0, len(bands_history['GSR'])-60), len(bands_history['GSR'])))
                           if len(history) > 30:
                               rng = max(history) - min(history)
                               if rng > 0.2:
                                   ui_refs['txt_motion_overlay'].set_text(f"UNSTEADY (Rng={rng:.2f})")
                               else:
                                   ui_refs['txt_motion_overlay'].set_text("")
                           else:
                               ui_refs['txt_motion_overlay'].set_text("")
                           
                           motion_lock_active = False # Reset state when expired
                       
             motion_prev_time = now
             motion_prev_ta = latest_gsr_ta
             
             # Init prev if needed
             if blowdown_prev_ta == 0.0: blowdown_prev_ta = latest_gsr_ta
             
             # Calculate Frame Delta for Path Efficiency
             frame_delta = abs(latest_gsr_ta - blowdown_prev_ta)
             
             # 1. Track Peaks (Pulse-to-Pulse / Continuous Rise)
             if latest_gsr_ta > blowdown_peak_ta:
                 blowdown_peak_ta = latest_gsr_ta
                 blowdown_peak_time = time.time() # [NEW] Capture time
                 blowdown_triggered = False # Reset trigger on rise
                 blowdown_path_len = 0.0 # Reset path
             else:
                 # Accumulate Path (only on drops/holds)
                 blowdown_path_len += frame_delta

             blowdown_prev_ta = latest_gsr_ta 

             # 2. Check Drop
             # If we haven't triggered yet, check magnitude
             # [REQ] CHECK GLOBAL LOCK
             if not blowdown_triggered and time.time() > motion_lock_expiry:
                 drop = blowdown_peak_ta - latest_gsr_ta
                 # [REQ] Warmup Check (Ignore first 3s)
                 is_warmed_up = (time.time() - session_start_time) > 3.0
                 
                 if drop >= 0.2 and is_warmed_up:
                     # [NEW] Motion Rejection Logic
                     # Efficiency = NetDrop / TotalPath
                     # If jagged, Path >> Drop -> Low Efficiency.
                     # If smooth, Path ~= Drop -> Efficiency ~ 1.0.
                     
                     efficiency = drop / (blowdown_path_len if blowdown_path_len > 0 else 1)
                     duration = time.time() - blowdown_peak_time
                     
                     is_valid = True
                     reason = ""
                     
                     if efficiency < 0.90: 
                         is_valid = False
                         reason = f"JAGGED (Eff={efficiency:.2f})"
                     elif duration < 0.1:
                         is_valid = False
                         reason = f"SPIKE (Dur={duration:.2f}s)"
                     # [REQ] Max Duration Selection (Sudden drops only)
                     elif duration > 2.0:
                         is_valid = False
                         reason = f"DRIFT (Dur={duration:.2f}s)"
                         
                     if is_valid:
                         active_event_label = f"TA_BLOWDOWN - DROP: {drop:.4f}"
                         blowdown_triggered = True # Prevent spam until next rise
                         # [NEW] Add to visual events (Peak Time, Drop Time)
                         blowdown_events.append((blowdown_peak_time, time.time()))
                     else:
                          # Log rejection (Optional, maybe debug only)
                          # active_event_label = f"REJECT_BLOWDOWN - {reason}" # Can be noisy
                          # Reset peak to prevent re-triggering on same artifact? 
                          # No, artifact might pass later if it smooths out? 
                          # If we reject 0.2, maybe we wait for 0.3?
                          # Actually, if we reject, we usually want to ignore this "Peak-Drop" pair?
                          # For now, just don't trigger. 
                          pass
             
             eff_win = get_effective_window()
             min_p = GSR_CENTER_VAL - (0.625 * eff_win)
             max_p = GSR_CENTER_VAL + (0.375 * eff_win)
             
             if latest_gsr_ta < min_p or latest_gsr_ta > max_p:
                 # Only if valid signal
                 if latest_gsr_ta > 0.1:
                      # [REQ] Motion Lock (Anti-Squirm)
                      # If signal is jumping (Range > 0.2 over last 1s), DO NOT Rescale/Recenter.
                      # This prevents "crazy" sensitivity during motion artifacts.
                      history = list(itertools.islice(bands_history['GSR'], max(0, len(bands_history['GSR'])-60), len(bands_history['GSR'])))
                      is_steady = True
                      if len(history) > 30:
                           rng = max(history) - min(history)
                           if rng > 0.2: is_steady = False
                      
                      if is_steady and time.time() > motion_lock_expiry:
                           if latest_gsr_ta < min_p: active_event_label = "DROP_TA_RESET"
                           else: active_event_label = "RISE_TA_RESET"
                           update_gsr_center(latest_gsr_ta)
                      else:
                           # Optional: Log block? or just ignore.
                           # active_event_label = "MOTION - AUTO-CENTER BLOCKED"
                           pass
        
        artists = []
        if current_view == 'main':
             # [NEW] Render Blowdown Markers (Scrolling Vertical Lines)
             # [FIX] Do NOT block during calib_mode. Allow markers to scroll off.
             if 'line_blowdown' in ui_refs:
                 bd_x = []
                 bd_y = []
                 now = time.time()
                 valid_events = []
                 # Graph Logic:
                 # HISTORY_LEN = 600 samples.
                 # Interval = 50ms (20Hz) [Animation Loop] BUT Data Appends at 60Hz [Reader Loop]
                 # X-Axis: 0 (Oldest) -> 600 (Newest).
                 # Age (seconds) * 60 = Age (samples).
                 # Position = 600 - Age_Samples.
                 samples_per_sec = 60.0 
                 window_samples = HISTORY_LEN
                 
                 # global blowdown_events # [FIX] Redundant, already declared at top of loop
                 for (t_peak, t_drop) in blowdown_events:
                     age_peak = now - t_peak
                     age_drop = now - t_drop
                     
                     x_peak = window_samples - (age_peak * samples_per_sec)
                     x_drop = window_samples - (age_drop * samples_per_sec)
                     
                     # Check visibility (Right edge is 500, Left is 0)
                     # Keep events until they scroll off left side (-10)
                     if x_drop > -10:
                         valid_events.append((t_peak, t_drop))
                         
                         # Draw Peak Line
                         if -10 <= x_peak <= window_samples + 1:
                             bd_x.extend([x_peak, x_peak, np.nan])
                             bd_y.extend([-5, 105, np.nan])
                         
                         # Draw Drop Line
                         if -10 <= x_drop <= window_samples + 1:
                             bd_x.extend([x_drop, x_drop, np.nan])
                             bd_y.extend([-5, 105, np.nan])
                             
                 blowdown_events = valid_events # Prune old
                 ui_refs['line_blowdown'].set_data(bd_x, bd_y)
                 artists.append(ui_refs['line_blowdown'])

             # [FIX] Add txt_gsr_status, txt_win_val
             
             # [NEW] Calibration Logic (Event-Driven)


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
                             # [REQ] Stability Gate: Wait 2.0s AND Steady Signal
                             # Calculate Steadiness (Range of last 1.0s / 60 samples)
                             history = list(itertools.islice(bands_history['GSR'], max(0, len(bands_history['GSR'])-60), len(bands_history['GSR'])))
                             is_steady = False
                             rng = 1.0
                             if len(history) > 30:
                                 rng = max(history) - min(history)
                                 # [MOD] Relaxed threshold from 0.02 to 0.05
                                 if rng < 0.05: is_steady = True
                             
                             # [FIX] Bypass Steadiness Check if we already have a Large Drop (Early Squeeze)
                             has_large_drop = (calib_base_ta - latest_gsr_ta) > 0.05
                             
                             if elapsed < 2.0 or (not is_steady and not has_large_drop):
                                  msg = f"CALIB {calib_step}/4: RELAX HAND...\n(Steadiness: {rng:.3f} / 0.05)"
                                  
                                  # [REQ] Hysteresis Tracking:
                                  # 1. Follow Rises (Recovery)
                                  if latest_gsr_ta > calib_base_ta:
                                      calib_base_ta = latest_gsr_ta
                                  else:
                                      # 2. Follow Small Drops (Wobble/Drift) but HOLD potential calibration drops
                                      if (calib_base_ta - latest_gsr_ta) < 0.05:
                                          calib_base_ta = latest_gsr_ta
                                  # Else: We Hold 'calib_base_ta' at the peak, assuming this is a Squeeze.
                             else:
                                  # [NEW] Log Gate Open for Analysis
                                  if calib_phase == 0: # One-time log on transition
                                      # We don't want to flood log, but this block runs every frame until phase changes.
                                      # Actually phase changes below immediately.
                                      pass
                                      
                                  msg = f"CALIB {calib_step}/4: SQUEEZE SENSOR\n(WAITING FOR DROP)"
                                  if latest_gsr_ta < calib_base_ta - 0.02: # Trigger Drop
                                      active_event_label = f"RELAX_GATE_OPEN - Rng:{rng:.3f}" # Log the state we came from
                                      calib_phase = 1
                                      calib_min_ta = latest_gsr_ta
                                      calib_start_time = time.time() # Reset timer for safety
                                      active_event_label = f"SQUEEZE_{calib_step}_DROP"
                        elif calib_phase == 1: # TRACKING DROP
                             msg = f"CALIB {calib_step}/4: SQUEEZE..."
                             if latest_gsr_ta < calib_min_ta:
                                 calib_min_ta = latest_gsr_ta
                             
                             # [Clean] Removed Mid-Drop Auto-Scale per user request.
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
                             
                                 # [REQ] Check Drop Ratio with current Sensitivity
                                 current_ratio = total_drop / BASE_SENSITIVITY
                                 
                                 # Rules: 
                                 # If Ratio > 0.63 (Drop takes up >63% of screen) -> Too Sensitive -> Expand Window
                                 # If Ratio < 0.40 (Drop takes up <40% of screen) -> Too Dull -> Shrink Window
                                 # Target: Drop = 50% of Window (Ratio = 0.50)
                                 # New Window = Drop / 0.50
                                 
                                 if current_ratio > 0.63 or current_ratio < 0.40:
                                     target_win = total_drop / 0.50
                                     target_win = max(0.05, min(50.0, target_win))
                                     
                                     # [REQ] Squeeze Stability Safeguard (Steps 2 & 3)
                                     # If change is drastic (>50%), IGNORE IT.
                                     should_update = True
                                     if calib_step > 1:
                                          pct_change = abs(target_win - BASE_SENSITIVITY) / BASE_SENSITIVITY
                                          if pct_change > 0.50:
                                               should_update = False
                                               log_msg(f"Calib {calib_step}: Drastic Change ({pct_change*100:.1f}%) -> IGNORED")
                                               active_event_label = f"SQUEEZE_{calib_step}_IGNORED_DRASTIC"
                                     
                                     drop_val = calib_base_ta - calib_min_ta
                                     
                                     if should_update:
                                          BASE_SENSITIVITY = target_win
                                          log_msg(f"Calib {calib_step}: Ratio={current_ratio:.2f} (OOB) -> Resized Base={BASE_SENSITIVITY:.2f}")
                                          active_event_label = f"SQUEEZE_{calib_step}_RESIZED - DROP: {drop_val:.4f}"
                                 else:
                                     drop_val = calib_base_ta - calib_min_ta
                                     log_msg(f"Calib {calib_step}: Ratio={current_ratio:.2f} (OK) -> Keep Base={BASE_SENSITIVITY:.2f}")
                                     active_event_label = f"SQUEEZE_{calib_step}_OK - DROP: {drop_val:.4f}"

                                 # txt_win_val.set_text(f"{1.0/BASE_SENSITIVITY:.2f}") # Update Display
                             
                                 # [REQ] Reset TA (Center Graph) after sensitivity change
                                 update_gsr_center(latest_gsr_ta)
                             
                                 # Move Next
                                 calib_step += 1
                                 calib_phase = 0
                                 calib_start_time = time.time()
                                 calib_base_ta = latest_gsr_ta 
                                 calib_min_ta = latest_gsr_ta
                                 
                                 # [CHECK] If this was step 4 (Breath), we might need to handle it in separate block?
                                 # No, Step 4 logic is below. This block handles 1,2,3.
                                 # Step 4 logic transitions to calib_phase 2 (Complete)
                                 # So we just ensure Step 4 Phase 2 does the final wrap up.

                        if ovl: ovl.set_text(msg)
                            
                            
                    # --- Step 4: BREATH ---
                    elif calib_step == 4:
                        if calib_phase == 0:
                            # [NEW] Timeout Check (30s)
                            if time.time() - calib_start_time > 30.0:
                                 log_msg("Calib Step 4 Timeout - Restarting")
                                 msg = "TIMEOUT - RESTARTING..."
                                 if ovl: ovl.set_text(msg); plt.draw(); plt.pause(1.0) # Show brief err
                                 start_calibration(None) # Restart
                                 return # Exit current loop
                            
                            # [REQ] Stability Gate
                            # Bypass Steadiness if Large Drop
                            has_large_drop = (calib_base_ta - latest_gsr_ta) > 0.05
                            
                            # Note: Step 4 doesn't have explicitly calculated 'rng' here, but relies on time mostly?
                            # Wait, Step 4 didn't calculate 'rng'. It only had Time Check.
                            # So Step 4 logic was just: if time < 2.0.
                            # But if I want to support Early Breath, I should just rely on the Hysteresis holding.
                            
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
                                 msg = "CALIB 4/4: DEEP BREATH\n(WAITING FOR DROP)"
                                 if latest_gsr_ta < calib_base_ta - 0.02:
                                    calib_phase = 1
                                    calib_min_ta = latest_gsr_ta
                                    calib_start_time = time.time() # [REQ] Start Stability Timer
                                    active_event_label = "BREATH_DROP"
                                    calib_step_start_time = time.time() # [NEW] Track total duration of this phase
                        elif calib_phase == 1:
                            msg = "CALIB 4/4: EXHALE / RELEASE..."
                            
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
                                    
                                    # [REQ] Final Sensitivity Resize based on Breath Drop
                                    # User wants this drop to be 50% of the window
                                    target_win = total_drop / 0.50
                                    target_win = max(0.05, min(50.0, target_win))
                                    BASE_SENSITIVITY = target_win
                                    
                                    update_gsr_center(latest_gsr_ta)
                                    active_event_label = f"BREATH_OK - FINAL_RESIZE - DROP: {total_drop:.4f}"
                                
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
                                    # [REQ] If mid-session re-calib?
                                    if is_recording:
                                        reset_count(None)
                                        session_start_ta = latest_gsr_ta
                                        
                                    calib_mode = False
                                    update_gsr_center(latest_gsr_ta) # [REQ] Trigger Pivot Set immediately
                                    if ovl: ovl.set_text("")
                                    log_msg("Calibration Complete")
                                    active_event_label = "CALIB_COMPLETE"

                        elif calib_phase == 3:
                             msg = "SESSION STARTED"
                             # [REQ] Start TA Counter Here (After Success)
                             if not counting_active: toggle_count(None)
                             
                             if time.time() - calib_start_time > 2.0:
                                  calib_mode = False
                                  update_gsr_center(latest_gsr_ta) # [REQ] Trigger Pivot Set immediately
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
             # [FIX] Removed bg_rect from blit to rely on full draw for correct Z-order layering with Buttons
             artists.extend([rec_text, txt_conn, txt_wear, txt_batt, system_line, txt_gsr_status]) 
             artists.extend(list(lines.values()))
             artists.extend([txt_ta_score, val_txt, txt_count_val, txt_win_val, txt_calib_overlay]) 
             
             # [FIX] Add Session Details to Artists
             if 'txt_sess_len' in ui_refs: artists.append(ui_refs['txt_sess_len'])
             if 'txt_sess_date' in ui_refs: artists.append(ui_refs['txt_sess_date'])
             if 'txt_sess_time' in ui_refs: artists.append(ui_refs['txt_sess_time'])
             
             if 'count_bg_rect' in ui_refs: artists.append(ui_refs['count_bg_rect'])
             
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
                 # Optional: Reset or Keep Last?
                 # If not recording, we can show "--"
                 pass 
             
             # [FIX] Reverted adding Button internal artists as it crashed processing.
             # Instead, we rely on fig.canvas.draw_idle() in the callbacks to refresh buttons.
             # The bg_rect should be enough for the background color change.
             # The button labels might lag only if the blit loop overwrites them?
             # But since they are on separate axes (reg_ax), blitting the main axes artists shouldn't clear them?
             # Wait, blit=True usually only updates the axes returned.
             # If Buttons are on DIFFERENT axes (ax_count_toggle), they are not involved in main_view_axes blitting?
             # Ah, FuncAnimation blits the *Figure* or specific artists?
             # It seems to blit everything returned.
             # If I don't return them, they don't get updated in the animation frame?
             # But they are static unless changed.
             # The crash was likely due to b.ax.patch not having the standard axes link expected by animation?
             # Or b.ax is None? No.
             # Safest: Don't add button internals. Just rely on draw_idle() for state changes.

        elif current_view == 'settings':
             artists.extend([bar_level[0], ui_refs['text_level'], log_text, system_line])

        return artists

    def on_close(event):
        global app_running
        app_running = False
        save_config() 
        print("Window Closed. Cleaning up...")

    fig.canvas.mpl_connect('close_event', on_close)

    try:
        # [OPT] Restored FPS to 50 (Interval 20ms) for smoothness, relying on Blit+Decimation for perf
        # [FIX] Disabled Blitting to resolve Button Visibility issues (Z-Order is respected in full draw)
        ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False) 
        
        try:
             audio_handler.sync_audio_stream('main') 
        except Exception as e:
             print(f"Audio sync error: {e}")
        
        
    # [FIX] Force UI updates one last time before show
        update_boost_ui()
        plt.draw()

        plt.show()
    except Exception as e:
        import traceback
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        app_running = False
        print("Stopping Bluetooth...")
        if f_gsr: f_gsr.close()
        if f_trend: f_trend.close()
        # [NEW] Stop HRM and Close File
        if hrm_sensor:
             try: hrm_sensor.stop()
             except: pass
        if f_hrm: 
             try: f_hrm.close()
             except: pass
             
        # [FIX] Stop Audio Stream using Handler
        if 'audio_handler' in globals() and audio_handler.audio_stream:
             audio_handler.audio_stream.close()
             
        t.join(timeout=2.0)
        print("Shutdown Complete.")
