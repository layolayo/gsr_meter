
import threading
import collections
import csv
import time
import math
import queue
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as st
import matplotlib
matplotlib.use('TkAgg') # Enforce TkAgg for compatibility with Tkinter popups
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
# [FIX] Unbind 'q' from default quit to allow custom handling
plt.rcParams['keymap.quit'] = ['ctrl+w', 'cmd+w'] # Removed 'q'
from datetime import datetime
import os
import json
import hid
from typing import Optional, Any
from modules.audio_handler import AudioHandler
# from modules.ant_driver import AntHrvSensor
# from modules.hrv_state_analyzer import HRVStateAnalyzer
# Pattern Rec
from modules.gsr_patterns import GSRPatterns 
from modules.session_viewer import SessionViewer # [NEW]

# File Naming
CONFIG_FILE = "v44_config.json"

# --- GSR SETTINGS ---
VENDOR_ID = 0x1fc9
PRODUCT_ID = 0x0003
V_SOURCE = 6.371
R_REF = 83.0

# --- GLOBAL VARIABLES ---
latest_gsr_ta = 0.0
GSR_CENTER_VAL = 3.0
GSR_CENTER_TARGET = 3.0 # [NEW] Target center for smooth dampening
# BASE_SENSITIVITY removed in favor of LOG_WINDOW_HEIGHT
LOG_WINDOW_HEIGHT = 0.05   # [MOD] Much higher starting sensitivity (approx 13.3x linear)
ZOOM_COEFFICIENT = 1.0     
CALIB_PIVOT_TA = 2.0
active_event_label = ""   
#latest_d_ta = 0.0 
last_stable_window = 0.125 # [MOD]
gsr_capture_queue = collections.deque(maxlen=100)
# [NEW] Manual Blit Globals
graph_bg = None 
bg_scores = None
bg_count  = None
bg_detail = None
bg_status = None
bg_sens   = None
# [NEW] Process Integration
from process_runner import ProcessRunner

bg_info = None # [FIX] Background for System Line
first_run_center = False # [FIX] Init Auto-Center flag
# bg_grid = None # [NEW] Bio-Grid BG
# bg_grid_txt = None # [NEW] Bio-Grid Text BG

# [NEW] Dynamic Vector Grid Analysis
grid_hist_ta = collections.deque(maxlen=150) # 3s @ 50Hz
# grid_hist_hrv = collections.deque(maxlen=150) # 3s @ 50Hz
# grid_hist_hr = collections.deque(maxlen=150) # 3s @ 50Hz
prev_quad = None # [NEW] Track Quadrant State for Transitions

# --- HRM GLOBALS ---
latest_hr = 0
latest_hrv = 0.0
# hrm_status = "Init"
# hrm_sensor = None
# hrv_analyzer = None # [NEW]
# latest_hrm_state = None # [NEW]
# txt_hr_val = None
# txt_hrv_val = None
gsr_patterns = None # [NEW] Engine
current_pattern = "IDLE"


# [NEW] Global Motion Detector (Velocity Filter)
motion_lock_expiry = 0.0
motion_prev_ta = 0.0
motion_prev_time = 0.0
motion_lock_active = False # [NEW] State tracker for logging
session_start_time = time.time() # [NEW] Track app start for Warmup

# [FIX] Missing Globals Initialized
calib_step_start_time = 0.0
notes_filename = None
current_state = "Init"
ignore_ui_callbacks = False
headset_on_head = False
# dead globals removed
pattern_hold_until = 0.0

# [NEW] Process State Globals
process_runner = None
active_process_name = None
active_process_data = None
process_step_idx = -1
process_waiting_for_calib = False
process_waiting_for_input = False
process_ending_phase = False
process_in_closing_phase = False # [NEW]
current_q_set = ""
current_q_id = ""
current_q_text = ""

# [NEW] User Management Globals
current_user_id = None
current_user_name = "Guest"
USERS_FILE = "users.json"
user_manager = None # Initialized later



# Global State for Assessment Selection
pending_assessment_selection = {}

def format_elapsed(total_seconds):
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    fraction = total_seconds - int(total_seconds)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{int(fraction * 100000):05}"


def get_effective_window():
    global last_stable_window, motion_lock_expiry    

    # Motion Safety: If Locked, FREEZE sensitivity at last known good value.
    if time.time() < motion_lock_expiry:
         return last_stable_window

    # Log scaling is "Set and Forget". We return the base log window.
    val = LOG_WINDOW_HEIGHT * ZOOM_COEFFICIENT
    
    last_stable_window = val
    return val

# --- STATE ---
# 60Hz * 10s = 600 samples (60Hz Plotting) -> Increased to 800 to ensure full coverage
HISTORY_LEN = 800
bands_history = {k: collections.deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN) for k in ['GSR']}

log_messages = collections.deque(["System Ready"], maxlen=10)
app_running = True
counting_active = False # TA Counter State
ta_accum = 0.0          # TA Counter Total
is_recording = False
last_motion_log = 0.0
samples_since_contact = 0
device_battery_level = "--"
ui_update_queue = queue.Queue()
command_queue = queue.Queue()

# Logic Settings
event_detected = False
total_samples_recorded = 0

# Control Defaults
global_percent = 20

# CSV Handles
f_gsr = None
writer_gsr: Optional[Any] = None
recording_start_time_obj: Optional[datetime] = None

# --- AUDIO RECORDING STATE ---
audio_filename = None


ZOOM_POSITIONS = [
    0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 
    1.2, 1.4, 1.6, 1.8, 2.0,
]

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
        self.samples_to_process = 0.0 
        self.last_flush_time = 0  


    def run(self):
        while self.running:
            try:
                global latest_gsr_ta, current_calm_score
                h = hid.device()
                h.open(VENDOR_ID, PRODUCT_ID)
                h.set_nonblocking(1) 
                self.connected = True
                print("[GSR] Connected (High Speed Mode)")

                while self.running and self.connected:
                    try:
                        data = None
                        while True: # Clear buffer loop
                            try:
                                d = h.read(64)
                                if d: data = d
                                else: break
                            except Exception: break
                        
                        if not data:
                            time.sleep(0.005) 
                            continue
                        
                        if len(data) >= 4 and data[0] == 0x01:
                            raw_val = (data[2] << 8) | data[3]
                            voltage = raw_val / 10000.0
                            
                            if voltage >= (V_SOURCE - 0.005):
                                ohms = 999999.9
                            else:
                                try:
                                    ohms = (voltage * R_REF) / (V_SOURCE - voltage)
                                except Exception:
                                    ohms = 999999.9
                            try:
                                ta = (ohms * 1000 / (ohms * 1000 + 21250)) * 5.559 + 0.941
                            except Exception:
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
                                      global CALIB_PIVOT_TA, active_event_label, motion_lock_expiry, current_pattern, ta_accum, recording_start_time_obj
                                      note = active_event_label
                                      if note: active_event_label = ""
                                      
                                      is_motion = 1 if time.time() < motion_lock_expiry else 0
                                      # Calc Elapsed
                                      elapsed_str = "00:00:00.00000"
                                      if recording_start_time_obj:
                                           elapsed_val = (datetime.now() - recording_start_time_obj).total_seconds()
                                           elapsed_str = format_elapsed(elapsed_val)
                                           
                                      # [NEW] Enhanced Logging with Question Metadata
                                      log_ta = math.log10(max(0.01, self.current_ta))
                                      # Access globals safely
                                      q_set = current_q_set if 'current_q_set' in globals() else ""
                                      q_id = current_q_id if 'current_q_id' in globals() else ""
                                      q_text = current_q_text if 'current_q_text' in globals() else ""
                                      
                                      writer_gsr.writerow([ts_now, elapsed_str, f"{self.current_ta:.5f}", f"{ta_accum:.5f}", f"{GSR_CENTER_VAL:.3f}", f"{1.0/win:.3f}", f"{win:.3f}", is_motion, f"{CALIB_PIVOT_TA:.3f}", note, current_pattern, f"{log_ta:.5f}", q_set, q_id, q_text])
                                 
                             except Exception: pass

                        # Periodic Flush to prevent IO Stalls (Every 1.0s)
                        if is_recording and (time.time() - self.last_flush_time) > 1.0:
                            try:
                                if f_gsr: f_gsr.flush()
                                self.last_flush_time = time.time()
                            except Exception: pass

                        # GRAPH HISTORY UPDATE (Master Clock = 60Hz)
                        try:
                             # 1. GSR Value
                             # Store RAW TA for dynamic scaling in main loop
                             bands_history['GSR'].append(self.current_ta)
                        except Exception: pass
                        
                        time.sleep(0.0166) # ~60Hz Pacing 

                    except Exception as loop_e:
                        print(f"[GSR] Loop Skip/Disconnect: {loop_e}")
                        self.connected = False
                        break # Break inner loop to trigger reconnect
            
            except Exception as e:
                if self.connected: # Only log if we were previously connected or it's a new attempt
                     print(f"[GSR] Connection Error: {e} - Retrying in 1s...")
                self.connected = False
                time.sleep(1.0)
            
            finally:
                try:
                    if 'h' in locals(): h.close()
                except: pass

    def stop(self):
        self.running = False

def log_msg(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    clean_msg = str(msg).strip()
    print(f"[{timestamp}] {clean_msg}")
    log_messages.append(f"[{timestamp}] {clean_msg}")
    
    # Update UI directly if available
    target = None
    if 'ui_refs' in globals():
         if 'log_text' in ui_refs: target = ui_refs['log_text']
         
    if target:
        try: target.set_text(clean_msg)
        except Exception: pass
        
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
        # vis_state removed (unused)
        # We need to access the 'check' widget or 'lines' to know what is visible.
        # Solution: Use a global `active_graph_lines` list or similar.
        # Let's inspect line 700 area again.
        
        if 'user_manager' in globals() and user_manager and current_user_id:
             user_manager.update_settings(current_user_id, {
                 'gsr_center': GSR_CENTER_VAL,
                 'log_window': LOG_WINDOW_HEIGHT,
                 'zoom_coeff': ZOOM_COEFFICIENT,
                 'voice_tld': process_runner.voice_tld if process_runner else "co.uk",
                 'voice_gender': process_runner.voice_gender if process_runner else "Female",
                 'audio_enabled': process_runner.audio_enabled if process_runner else True
             })
             
        cfg = {
            'mic_name': audio_handler.current_mic_name if 'audio_handler' in globals() else "Default",
            'mic_gain': audio_handler.current_mic_gain if 'audio_handler' in globals() else 3.0,
            'mic_rate': audio_handler.current_mic_rate if 'audio_handler' in globals() else None
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

        print(f"[Config] Loaded settings.")
    except Exception as e:
        print(f"[Config] Load Error: {e}")

# --- MANUAL VIEWER COMPONENT ---
from modules.manual_viewer import ManualViewer

# Global Manual Instance
manual_viewer = ManualViewer(
    title="Emergent Knowledge Manual 🧠",
    pages=[
        ("Master Index", "manual/index.md", "#2c3e50"),
        ("Introduction", "manual/introduction.md", "#2980b9"),
        ("Device Calibration", "manual/calibration.md", "#e67e22"),
        ("Graph Patterns", "manual/graph_patterns.md", "#2980b9"),
        ("Training Drills", "manual/training_drills.md", "#27ae60"),
        ("Advanced Drills", "manual/advanced_drills.md", "#c0392b"),
        ("Assessment Lists", "manual/assessment_lists.md", "#8e44ad"),
        ("Pre-Session Processes", "manual/pre_session.md", "#1abc9c"),
        ("Training Processes", "manual/processes.md", "#d35400"),
        ("Method One", "manual/method_one.md", "#f39c12"),
        ("Advanced Processes", "manual/advanced_processes.md", "#f39c12"),
        ("Technical & Troubleshooting", "manual/technical_guide.md", "#2980b9")
    ]
)

class UserManager:
    def __init__(self):
        self.users = {} # ID -> Settings Dict (must include "name")
        self.last_user_id = None
        self.default_settings = {
            "name": "Guest",
            "voice_tld": "co.uk",
            "voice_gender": "Female",
            "audio_enabled": True,
            "gsr_center": 3.0,
            "log_window": 0.05,
            "zoom_coeff": 1.0
        }
        self.load()

    def load(self):
        if os.path.exists(USERS_FILE):
            try:
                with open(USERS_FILE, 'r') as f:
                    data = json.load(f)
                    raw_users = data.get("users", {})
                    
                    # Migration Logic
                    if isinstance(raw_users, list):
                        # Old list format: ["User1", "User2"]
                        self.users = {}
                        for i, name in enumerate(raw_users):
                            u_id = i + 1
                            s = self.default_settings.copy()
                            s["name"] = name
                            self.users[u_id] = s
                    elif isinstance(raw_users, dict):
                        # Check if keys are names (old dict format) or IDs (new format)
                        sample_key = next(iter(raw_users.keys())) if raw_users else None
                        if sample_key and not sample_key.isdigit():
                            # Old Name-indexed dict: {"Name": {settings}}
                            self.users = {}
                            for i, (name, settings) in enumerate(raw_users.items()):
                                u_id = settings.get("id", i + 1)
                                s = settings.copy()
                                s["name"] = name
                                self.users[u_id] = s
                        else:
                            # New ID-indexed dict: {"1": {name: "Name", settings}}
                            self.users = {int(k): v for k, v in raw_users.items()}
                    
                    self.last_user_id = data.get("last_user_id")
                    if not self.last_user_id and "last_user" in data:
                        # Migrate last_user (name) to ID
                        name = data["last_user"]
                        for uid, udata in self.users.items():
                            if udata.get("name") == name:
                                self.last_user_id = uid
                                break
            except Exception as e:
                print(f"[User] Load Error: {e}")
        
        if not self.users:
            self.users = {1: self.default_settings.copy()}
            self.last_user_id = 1

    def save(self):
        try:
            with open(USERS_FILE, 'w') as f:
                json.dump({"users": self.users, "last_user_id": self.last_user_id}, f, indent=4)
        except Exception as e:
            print(f"[User] Save Error: {e}")

    def add_user(self, name):
        # Find next ID
        new_id = max(self.users.keys()) + 1 if self.users else 1
        s = self.default_settings.copy()
        s["name"] = name
        self.users[new_id] = s
        self.save()
        return new_id

    def set_last_user(self, u_id):
        self.last_user_id = u_id
        self.save()
        
    def get_settings(self, u_id):
        if u_id in self.users:
            return self.users[u_id]
        return self.default_settings.copy()
        
    def update_settings(self, u_id, settings):
        if u_id in self.users:
            self.users[u_id].update(settings)
            self.save()

    def get_name(self, u_id):
        return self.users.get(u_id, {}).get("name", "Unknown")

# [NEW] User Selection Dialog
from tkinter import simpledialog

def show_user_selection_dialog():
    global current_user_id, current_user_name, user_manager
    if user_manager is None:
        user_manager = UserManager()
    
    root = tk.Tk(); root.withdraw()
    dlg = tk.Toplevel(root)
    dlg.title("Select User")
    dlg.geometry("300x400")
    dlg.configure(bg='#2b2b2b')
    
    tk.Label(dlg, text="Who is training?", bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold')).pack(pady=10)
    
    frame_list = tk.Frame(dlg, bg='#2b2b2b')
    frame_list.pack(fill=tk.BOTH, expand=True, padx=10)
    
    lb = tk.Listbox(frame_list, bg='#333', fg='white', font=('Arial', 11), height=10)
    # Map Display Name -> ID
    name_to_id = {}
    for u_id, u_data in user_manager.users.items():
        u_name = u_data.get("name", f"User {u_id}")
        name_to_id[u_name] = u_id
    
    sorted_names = sorted(name_to_id.keys())
    for name in sorted_names:
        lb.insert(tk.END, name)
    lb.pack(fill=tk.BOTH, expand=True)
    
    # Pre-select last
    last_name = user_manager.get_name(user_manager.last_user_id)
    if last_name in sorted_names:
        try:
            idx = sorted_names.index(last_name)
            lb.selection_set(idx)
            lb.see(idx)
        except: pass
    
    id_selected = [None]
    
    def on_select():
        sel = lb.curselection()
        if not sel: return
        name = lb.get(sel[0])
        id_selected[0] = name_to_id.get(name)
        dlg.destroy()
        
    def on_add_new():
        new_name = simpledialog.askstring("New User", "Enter Name:", parent=dlg)
        if new_name and new_name.strip():
            name = new_name.strip()
            new_id = user_manager.add_user(name)
            # Refresh list
            lb.delete(0, tk.END)
            name_to_id.clear()
            for u_id, u_data in user_manager.users.items():
                name_to_id[u_data["name"]] = u_id
            
            nonlocal sorted_names
            sorted_names = sorted(name_to_id.keys())
            for n in sorted_names: lb.insert(tk.END, n)
            
            idx = sorted_names.index(name)
            lb.selection_set(idx)
            lb.see(idx)

    btn_frame = tk.Frame(dlg, bg='#2b2b2b')
    btn_frame.pack(pady=10, fill=tk.X)
    
    tk.Button(btn_frame, text="Select", command=on_select, bg='#004400', fg='white', width=10).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="New User", command=on_add_new, bg='#004488', fg='white', width=10).pack(side=tk.RIGHT, padx=10)
    
    # Center
    dlg.update_idletasks()
    x = (dlg.winfo_screenwidth() // 2) - (dlg.winfo_width() // 2)
    y = (dlg.winfo_screenheight() // 2) - (dlg.winfo_height() // 2)
    dlg.geometry(f"+{x}+{y}")
    
    dlg.transient()
    dlg.grab_set()
    dlg.wait_window()
    root.destroy()
    
    if id_selected[0] is not None:
        current_user_id = id_selected[0]
        user_manager.set_last_user(current_user_id)
    else:
        # Fallback to last user if closed without selection
        current_user_id = user_manager.last_user_id
        
    current_user_name = user_manager.get_name(current_user_id)
    apply_user_settings(current_user_id, user_manager)

def apply_user_settings(u_id, manager):
    """Applies per-user settings to global variables and ProcessRunner."""
    global LOG_WINDOW_HEIGHT, GSR_CENTER_VAL, ZOOM_COEFFICIENT, process_runner
    settings = manager.get_settings(u_id)
    name = settings.get("name", "Unknown")
    
    LOG_WINDOW_HEIGHT = settings.get("log_window", 0.05)
    GSR_CENTER_VAL = settings.get("gsr_center", 3.0)
    ZOOM_COEFFICIENT = settings.get("zoom_coeff", 1.0)
    
    if process_runner:
        process_runner.voice_tld = settings.get("voice_tld", "co.uk")
        process_runner.voice_gender = settings.get("voice_gender", "Female")
        process_runner.audio_enabled = settings.get("audio_enabled", True)
        
        # [NEW] Sync Settings UI Buttons
        if 'accent_btns' in ui_refs:
            for tld, btn in ui_refs['accent_btns'].items():
                btn.color = '#444444' if tld != process_runner.voice_tld else '#0066aa'
        
        if 'gender_btns' in ui_refs:
             for gen, btn in ui_refs['gender_btns'].items():
                  btn.color = '#444444' if gen != process_runner.voice_gender else '#006688'

        if 'btn_audio_toggle' in ui_refs:
            btn = ui_refs['btn_audio_toggle']
            btn.label.set_text("Audio: OFF" if not process_runner.audio_enabled else "Audio: ON")
            btn.color = '#660000' if not process_runner.audio_enabled else '#006600'
    
    print(f"[User] Applied settings for '{name}': Accent={settings.get('voice_tld')}, Gender={settings.get('voice_gender')}, Audio={'ON' if settings.get('audio_enabled') else 'OFF'}")
    
    # Update UI if needed
    if 'txt_sess_user' in ui_refs:
        ui_refs['txt_sess_user'].set_text(f"User: {name}")
    
    update_zoom_ui()



def map_zoom_to_nearest(zoom_val, slider=None):
    """
    Snap a zoom coefficient to the nearest allowed position.
    Parameters
    ----------
    zoom_val : float
        The raw zoom coefficient (e.g. from the slider or arrow keys).
    slider : matplotlib.widgets.Slider, optional
        If supplied, the slider thumb will be moved to the snapped value.
    Returns
    -------
    float
        The nearest zoom position from ``ZOOM_POSITIONS``.
    """
    # Find the nearest entry in ZOOM_POSITIONS
    nearest = min(ZOOM_POSITIONS, key=lambda x: abs(x - zoom_val))
    # Update the Matplotlib slider if we have one
    if slider is not None:
        slider.set_val(nearest)
    return nearest

def show_manual_popup(event=None):
    # Ensure this runs on the main thread or via a safe call
    # With Tkinter + Matplotlib this is usually OK if on same main thread
    manual_viewer.show()

if __name__ == "__main__":
    # Globals Init
    
    # Start GSR Thread
    gsr_thread = GSRReader()
    gsr_thread.start()

    pass 
    
    # --- GUI ---
    plt.rcParams['toolbar'] = 'None' # [NEW] Hide Matplotlib navigation controls
    fig = plt.figure(figsize=(15, 9), facecolor='#2b2b2b') 
    try: fig.canvas.manager.set_window_title("EK GSR Session Monitor (v42 Dark Viewer)")
    except Exception: pass 
    
    # [NEW] Start Maximized/Fullscreen (Linux Compatibility)
    try:
        mngr = plt.get_current_fig_manager()
        # TkAgg/Linux
        try: 
            mngr.window.attributes('-fullscreen', True)
            mngr.window.update() # [NEW] Force Tkinter update
        except: 
             try: 
                 mngr.window.state('zoomed') # Windows fallback
                 mngr.window.update()
             except: pass
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
        except Exception: pass
        
    audio_handler = AudioHandler(log_msg, update_audio_ui)
    


    
    # [FIX] Load Config EARLY so we know which Mic to probe, and correct GSR defaults
    load_config()

    # [NEW] Initialize Process Runner
    try:
        process_runner = ProcessRunner("processes.json")
        print(f"[Process] Loaded {len(process_runner.list_processes())} processes.")
    except Exception as e:
        print(f"[Process] Init Error: {e}")

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
        ax_p.set_xticks([]); ax_p.set_yticks([]); ax_p.set_facecolor('#333333')
        ax_p.text(0.05, 0.95, title, transform=ax_p.transAxes, ha='left', va='top', fontsize=9, fontweight='bold', color='#dddddd')
        return ax_p

    # === ROLLING LINE GRAPH (Main Area) ===
    ax_left_labels = reg_ax([0.00, 0.20, 0.05, 0.70], main_view_axes)
    ax_left_labels.set_xlim(0, 1)
    ax_left_labels.set_ylim(-5, 105) # Sync with graph Y
    ax_left_labels.set_axis_off()
    
    ax_graph = reg_ax([0.05, 0.20, 0.78, 0.70], main_view_axes)
    ax_graph.set_xlim(0, HISTORY_LEN)
    ax_graph.set_ylim(-5, 105)
    ax_graph.set_facecolor('#1e1e1e')
    ui_refs['ax_graph'] = ax_graph 
    
    ax_graph.set_title("GSR Monitor ~10s", fontsize=14, fontweight='bold', color='white')
    ax_graph.set_xticks([0,80,160,240,320,400,480,560,640,720,800])
    ax_graph.set_xticklabels(["10","9","8","7","6","5","4","3","2","1","0"])
    ax_graph.tick_params(colors='lightgray')
    for spine in ax_graph.spines.values():
        spine.set_edgecolor('#555555')
    
    # [FIX] Ensure Spines are ON TOP of the data line (Z=100)
    for spine in ax_graph.spines.values():
        spine.set_zorder(100)

    # Create lines
    lines = {}
   
    line, = ax_graph.plot([], [], lw=2, color='magenta', label='GSR')
    lines['GSR'] = line
    ax_graph.axhline(y=62.5, color='#CC5500', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    
    # [NEW] Unit Grid Lines (1 Unit = 20% of Window)
    # TA SET is at 62.5. Lines at +/- 20, 40.
    # 62.5 + 40 = 102.5 (2 Above)
    # 62.5 + 20 = 82.5 (1 Above)
    # 62.5 - 20 = 42.5 (1 Below)
    # 62.5 - 40 = 22.5 (2 Below)
    # 62.5 - 60 = 2.5  (3 Below)
    
    # Map Y-Pos to "Unit Offset" (1 Unit = 20pts)
    # 102.5 -> +2 Units
    # 82.5  -> +1 Unit
    # 42.5  -> -1 Unit
    # 22.5  -> -2 Units
    # 2.5   -> -3 Units
    unit_defs = [
        (102.5, 2), 
        (82.5, 1), 
        (42.5, -1), 
        (22.5, -2), 
        (2.5, -3)
    ]
    
    unit_texts = []
    
    for y_u, u_idx in unit_defs:
         ax_graph.axhline(y=y_u, color='#CC5500', linestyle=':', linewidth=1.0, alpha=0.5, zorder=0)
         # [NEW] Dynamic Label (In Left Axes)
         t = ax_left_labels.text(0.95, y_u, f"{u_idx:+d}", color='#CC5500', fontsize=8, alpha=0.7, 
                           ha='right', va='top', rotation=45)
         t.set_animated(True)
         ui_refs[f'txt_grid_{u_idx}'] = t
         unit_texts.append((t, u_idx))

    txt_ta_set_line = ax_left_labels.text(0.95, 62.5, f"TA: {GSR_CENTER_VAL:.2f}", 
                                    color='#CC5500', fontsize=8, fontweight='bold', 
                                    ha='right', va='top', rotation=45)
    ui_refs['txt_ta_set_line'] = txt_ta_set_line
    # [NEW] Manual Blit Background Capture
    def on_draw(_event):
        global graph_bg, bg_scores, bg_count, bg_detail, bg_status, bg_sens, bg_left_labels, bg_scale_panel, bg_system_panel
        if fig:
            if ax_left_labels: bg_left_labels = fig.canvas.copy_from_bbox(ax_left_labels.bbox)
            if ax_graph: graph_bg = fig.canvas.copy_from_bbox(ax_graph.bbox)
            if ax_scores: bg_scores = fig.canvas.copy_from_bbox(ax_scores.bbox)
            if ax_count_bg: bg_count = fig.canvas.copy_from_bbox(ax_count_bg.bbox)
            if ax_detail: bg_detail = fig.canvas.copy_from_bbox(ax_detail.bbox)
            if ax_status: bg_status = fig.canvas.copy_from_bbox(ax_status.bbox)
            if ax_w_val:  bg_sens   = fig.canvas.copy_from_bbox(ax_w_val.bbox)
            if ax_ctrl_bg: bg_scale_panel = fig.canvas.copy_from_bbox(ax_ctrl_bg.bbox)
            if 'ax_system_bg' in ui_refs: bg_system_panel = fig.canvas.copy_from_bbox(ui_refs['ax_system_bg'].bbox)
            
            # [BIO-GRID REMOVED]
            
    fig.canvas.mpl_connect('draw_event', on_draw)
    
    # [FIX] Set Animated (Exclude from Background)
    if 'txt_ta_set_line' in locals(): locals()['txt_ta_set_line'].set_animated(True)
    if line: line.set_animated(True)
    
    # [FIX] Restore Overlay Axis for Global Text
    ax_overlay = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax_overlay.set_xticks([]); ax_overlay.set_yticks([])
    ax_overlay.set_zorder(100) # Ensure on top
    
    # [NEW] Calibration Overlay Text (Ax Level)
    # [FIX] Re-parented to ax_graph to ensure it is centered on the GRAPH, not the window.
    txt_calib_overlay = ax_graph.text(0.5, 0.5, "", ha='center', va='center', fontsize=24, fontweight='bold', color='red', transform=ax_graph.transAxes)
    ui_refs['txt_calib_overlay'] = txt_calib_overlay 
    txt_calib_overlay.set_animated(True) # [FIX] Animated
    
    # [NEW] Motion Overlay Text (Center of Graph)
    txt_motion_overlay = ax_graph.text(0.5, 0.5, "", ha='center', va='center', fontsize=20, fontweight='bold', color='red', transform=ax_graph.transAxes)
    ui_refs['txt_motion_overlay'] = txt_motion_overlay
    txt_motion_overlay.set_animated(True) # [FIX] Animated

    # [NEW] Process Overlay Text (Top Center of Graph - Distinct from Calib)
    txt_process_overlay = ax_graph.text(0.5, 0.90, "", ha='center', va='top', fontsize=14, fontweight='bold', color='#FFCC00', transform=ax_graph.transAxes)
    ui_refs['txt_process_overlay'] = txt_process_overlay
    txt_process_overlay.set_animated(True)
    
    # [NEW] GSR Pattern Text (Bottom Center of Graph)
    txt_pattern = ax_graph.text(0.5, 0.02, "PATTERN: IDLE", transform=ax_graph.transAxes, ha='center', va='bottom', fontsize=14, fontweight='bold', color='gray', zorder=90)
    txt_pattern.set_animated(True) # [FIX] Prevent Ghosting
    ui_refs['txt_pattern'] = txt_pattern
    
    # === GSR CONTROLS PANEL ===
    r_ctrl = [0.835, 0.68, 0.13, 0.22] # [MOD] Height and Pos for Scale grouping
    ax_ctrl_bg = reg_ax(r_ctrl, main_view_axes)
    ax_ctrl_bg.set_facecolor('#333333')
    ax_ctrl_bg.set_xticks([]); ax_ctrl_bg.set_yticks([])
    # Border
    rect_ctrl_border = plt.Rectangle((0,0), 1, 1, transform=ax_ctrl_bg.transAxes, fill=False, ec='#555555', lw=2, clip_on=False)
    ax_ctrl_bg.add_patch(rect_ctrl_border)
    
    # --- 1. Title: GSR SCALE ---
    ax_scale_lbl = reg_ax([0.835, 0.86, 0.13, 0.03], main_view_axes)
    ax_scale_lbl.set_axis_off()
    ax_scale_lbl.text(0.5, 0.5, "GSR SCALE", ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    
    # --- 2. Zoom Level Slider Label ---
    ax_zoom_lbl = reg_ax([0.835, 0.825, 0.13, 0.025], main_view_axes)
    ax_zoom_lbl.set_axis_off()
    ax_zoom_lbl.text(0.5, 0.5, "Zoom Level", ha='center', va='center', fontsize=9, fontweight='bold', color='#cccccc')
    
    # [NEW] Slider Track
    y_zoom = 0.79 
    ax_zoom_track = reg_ax([0.845, y_zoom, 0.11, 0.025], main_view_axes)
    ax_zoom_track.set_facecolor('#444444')
    ax_zoom_track.set_xticks([]); ax_zoom_track.set_yticks([])
    ax_zoom_track.set_xlim(0, 1) # [NEW] Force exact coordinate mapping
    ax_zoom_track.margins(0)      # [NEW]
    ax_zoom_track.set_navigate(False) # [NEW] Disable standard pan/zoom
    ax_zoom_track.set_zorder(500) 
    # Slider line
    rect_track = plt.Rectangle((0, 0.45), 1, 0.1, transform=ax_zoom_track.transAxes, color='#888888')
    ax_zoom_track.add_patch(rect_track)
    # Slider thumb
    thumb_zoom, = ax_zoom_track.plot([0.5], [0.5], 'o', color='#00ff00', markersize=10, transform=ax_zoom_track.transAxes)
    
    # [NEW] Visual References (Tick & Labels)
    ax_zoom_track.axvline(0.5, color='#ffffff', lw=1, alpha=0.5, ymin=0, ymax=1) # Center Tick
    txt_z_left = ax_zoom_track.text(0.0, -0.6, "0.5x", transform=ax_zoom_track.transAxes, ha='left', va='top', fontsize=7, color='#888888')
    txt_z_mid  = ax_zoom_track.text(0.5, -0.6, "1x", transform=ax_zoom_track.transAxes, ha='center', va='top', fontsize=7, color='#ffffff')
    txt_z_right= ax_zoom_track.text(1.0, -0.6, "2.0x", transform=ax_zoom_track.transAxes, ha='right', va='top', fontsize=7, color='#888888')

    # [FIX] Throttled Blitting: Restore background once, draw dynamic artists, blit
    ui_refs['ax_zoom_track'] = ax_zoom_track
    ui_refs['rect_track'] = rect_track
    ui_refs['thumb_zoom'] = thumb_zoom
    ui_refs['txt_z_labels'] = [txt_z_left, txt_z_mid, txt_z_right] # For blitting

    def get_display_sens():
        # [MOD] Returns TOTAL Effective Sensitivity (Base * Zoom)
        w_eff = LOG_WINDOW_HEIGHT * ZOOM_COEFFICIENT
        if w_eff <= 0.0001: return 99.9
        return 1.0 / (w_eff * 1.5)

    def get_span_pct():
        # [NEW] Total screen height as a percentage of TA
        # Top/Bottom ratio is 10^W
        w_eff = LOG_WINDOW_HEIGHT * ZOOM_COEFFICIENT
        return (math.pow(10, w_eff) - 1.0) * 100.0

    def update_zoom_ui():
        # [MOD] Update Slider Position based on current ZOOM_COEFFICIENT
        try:
            x_norm = (math.log(1.0/ZOOM_COEFFICIENT, 2.0) / 2.0) + 0.5
            thumb_zoom.set_xdata([x_norm])
        except: pass
        
        # [MOD] Unified Display: SPAN: X% [Multiplier]
        # We use an integer for Span as requested.
        try:
             span_val = int(round(get_span_pct()))
             zoom_val = 1.0 / ZOOM_COEFFICIENT
             txt_win_val.set_text(f"SPAN: {span_val}%")
             txt_span_val.set_text(f"[{zoom_val:.1f}x]")
        except: pass
        
        # [FIX] REMOVED draw_idle() from main loop 
        # Redrawing is handled by final_blit()

    # [NEW] Event Handler for Zoom Slider (with Grabbed state)
    slider_grabbed = False

    def on_zoom_click(event):
        global slider_grabbed
        if event.button == 1:
            if event.name == 'button_press_event' and event.inaxes == ax_zoom_track:
                slider_grabbed = True
            
            if slider_grabbed and event.inaxes == ax_zoom_track:
                global ZOOM_COEFFICIENT
                inv = ax_zoom_track.transAxes.inverted()
                x_ax, y_ax = inv.transform((event.x, event.y))
                x_val = max(0, min(1, x_ax))
                
                # [MOD] Stepped Zoom (Snap to 0.1 Increments)
                # Mult = 2 ^ ((x-0.5)*2)
                mult_raw = math.pow(2.0, (x_val - 0.5) * 2)
                
                # Snap to specific 0.1 steps [0.5, 0.6 ... 2.0]
                steps = ZOOM_POSITIONS
                mult_stepped = min(steps, key=lambda s: abs(s - mult_raw))
                
                ZOOM_COEFFICIENT = 1.0 / mult_stepped
                update_zoom_ui()

    def on_zoom_release(event):
        global slider_grabbed
        if slider_grabbed:
            slider_grabbed = False

    def on_key_press(event):
        global ZOOM_COEFFICIENT, process_waiting_for_input, process_ending_phase
        # [NEW] Arrow Key Zoom Controls (Locked to ZOOM_POSITIONS)
        if event.key in ['left', 'right', 'up', 'down']:
            steps = ZOOM_POSITIONS
            current_mult = 1.0 / ZOOM_COEFFICIENT
            
            # Find nearest step index
            nearest = min(steps, key=lambda s: abs(s - current_mult))
            try:
                idx = steps.index(nearest)
            except ValueError:
                idx = steps.index(1.0) # Fallback

            # Right/Up increases multiplier (zooms in), Left/Down decreases
            direction = 1 if event.key in ['right', 'up'] else -1
            new_idx = max(0, min(len(steps)-1, idx + direction))
            new_mult = steps[new_idx]
            
            ZOOM_COEFFICIENT = 1.0 / new_mult
            update_zoom_ui()
        
        # [NEW] Process Input Hook
        elif process_waiting_for_input and event.key in [' ', 'enter']:
            process_waiting_for_input = False
            active_event_label = "PROCESS_USER_ADVANCE"
            
            if process_ending_phase:
                 finish_process_session()
            else:
                 advance_process_step()
            
        # [NEW] Spacebar auto-sets TA Center to current value
        elif event.key == ' ':
            if latest_gsr_ta > 0.1:
               update_gsr_center(latest_gsr_ta, reason="User Reset")

    def on_scroll(event):
        global ZOOM_COEFFICIENT
        if event.inaxes == ax_graph or event.inaxes == ax_zoom_track:
            steps = ZOOM_POSITIONS
            current_mult = 1.0 / ZOOM_COEFFICIENT
            nearest = min(steps, key=lambda s: abs(s - current_mult))
            try:
                idx = steps.index(nearest)
            except ValueError:
                idx = steps.index(1.0)

            # Scroll Up (event.step > 0) zooms in (increases multiplier)
            direction = 1 if event.step > 0 else -1
            new_idx = max(0, min(len(steps)-1, idx + direction))
            new_mult = steps[new_idx]

            ZOOM_COEFFICIENT = 1.0 / new_mult
            update_zoom_ui()

    fig.canvas.mpl_connect('button_press_event', on_zoom_click)
    fig.canvas.mpl_connect('button_release_event', on_zoom_release)
    fig.canvas.mpl_connect('motion_notify_event', on_zoom_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # --- 3. Sensitivity Value Display ---
    y_sens_val = 0.695 # [MOD] Positioned neatly in panel
    ax_w_val = reg_ax([0.85, y_sens_val, 0.10, 0.06], main_view_axes) # [MOD] Height 0.04 -> 0.06
    ui_refs['ax_w_val'] = ax_w_val
    ax_w_val.set_xticks([]); ax_w_val.set_yticks([])
    ax_w_val.set_facecolor('#333333') # [FIX] Ensure consistent background
    ax_w_val.set_zorder(100)
    # Clear visual border
    ax_w_val.add_patch(plt.Rectangle((0,0), 1, 1, transform=ax_w_val.transAxes, fill=False, ec='#555555', lw=1))
    
    # Value text (Split into two lines)
    txt_win_val = ax_w_val.text(0.5, 0.72, "", ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    txt_span_val = ax_w_val.text(0.5, 0.28, "", ha='center', va='center', fontsize=10, color='#aaaaaa')
    
    txt_win_val.set_animated(True)
    txt_span_val.set_animated(True)
    ui_refs['txt_win_val'] = txt_win_val 
    ui_refs['txt_span_val'] = txt_span_val 
    ui_refs['ax_w_val'] = ax_w_val
    
    update_zoom_ui() # Initial draw
    
             
    def start_calibration(event):
        global calib_mode, calib_step, calib_phase, calib_start_time, calib_base_ta, calib_min_ta, calib_vals, calib_step_start_time
        global CALIB_PIVOT_TA, ZOOM_COEFFICIENT
        
        # [MOD] Ensure we have a sane baseline if current is essentially 0
        global LOG_WINDOW_HEIGHT
        if LOG_WINDOW_HEIGHT < 0.01:
             LOG_WINDOW_HEIGHT = 0.125 
        
        # [NEW] Reset Zoom on Calibration
        ZOOM_COEFFICIENT = 1.0
        update_zoom_ui()
        
        calib_mode = True
        calib_phase = 0
        calib_step = 1
        calib_start_time = time.time()
        calib_vals = [] # [NEW] Median Buffer
        calib_base_ta = latest_gsr_ta
        calib_min_ta = latest_gsr_ta
        calib_vals = [] # Store drops
        
        log_msg(f"Calibration Started.")
        global active_event_label
        active_event_label = "CALIB_START"
        update_gsr_center(latest_gsr_ta)
           
    
    def rounded_step(val): return round(val * 1000) / 1000.0 # Round to nearest 0.001


    # [FIX] Moved TA SET to central box
    # val_txt defined in Scores Panel


    # [FIX] Added last_calib_ratio for debug
    last_calib_ratio = 0.0

    def update_gsr_center(val, force_pivot=False, reason="System"):
        global GSR_CENTER_VAL, GSR_CENTER_TARGET, ta_accum, calib_mode # [FIX] Added GSR_CENTER_TARGET
        global motion_lock_expiry 
        global CALIB_PIVOT_TA
        
        # [NEW] TA Counter Logic (Count Drops)
        if counting_active:
             # [REQ] Global Motion Block
             if time.time() > motion_lock_expiry:
                 diff = GSR_CENTER_TARGET - val  # [MOD] Compare against Target
                 # [FIX] Noise Floor (0.001): Block micro-jitter
                 if diff > 0.001 and not calib_mode: 
                     ta_accum += diff
                     print(f"[TA] Accumulating Drop: +{diff:.3f} (Total: {ta_accum:.2f})")
        
        if not calib_mode:
            if force_pivot:
                 CALIB_PIVOT_TA = max(0.1, val)
                 log_msg(f"Pivot FORCE Set to {CALIB_PIVOT_TA:.2f}")
            else:
                 if reason.startswith("User"):
                      CALIB_PIVOT_TA = max(0.1, val)
        
        GSR_CENTER_TARGET = val # [MOD] Update Target only
        if val_txt: val_txt.set_text(f"TA SET: {val:.2f}")

    # Keyboard Control
    
    # === TA & SCORES PANELS (Clean Re-implementation) ===
    # Moved UP to avoid overlapping System Info (Y=0.04)
    # Graph Bottom is now 0.20, so we have 0.04 to 0.20 to work with (Height 0.16)
    
    # 1. TA Counter Panel (Right Side of Left Block)
    r_count = [0.20, 0.015, 0.20, 0.12]
    ax_count_bg = reg_ax(r_count, main_view_axes)
    ax_count_bg.set_xticks([]); ax_count_bg.set_yticks([])
    
    # Background Patch (Dynamic Color)
    bg_count_rect = plt.Rectangle((0,0), 1, 1, transform=ax_count_bg.transAxes, color='#003300', ec='#555', lw=2, clip_on=False)
    bg_count_rect.set_zorder(0) # Patch at bottom of Axes
    ax_count_bg.add_patch(bg_count_rect)
    ui_refs['count_bg_rect'] = bg_count_rect
    
    # Text
    txt_count_val = ax_count_bg.text(0.5, 0.70, "TA Counter: 0.00", ha='center', va='center', fontsize=16, fontweight='bold', color='#66ff66')
    txt_count_val.set_zorder(10)
    txt_count_val.set_animated(True)
    
    # Buttons (Axes inside the panel area relative to Figure)
    ax_btn_start = reg_ax([0.23, 0.025, 0.07, 0.04], main_view_axes)
    ax_btn_reset = reg_ax([0.31, 0.025, 0.07, 0.04], main_view_axes)
    
    # Ensure Z-Order (Axes Level)
    ax_count_bg.set_zorder(1)
    ax_btn_start.set_zorder(100)
    ax_btn_reset.set_zorder(100)
    
    # Start/Reset Buttons
    
    ui_refs['btn_count'] = Button(ax_btn_start, 'Start', color='#004400', hovercolor='#006600') 
    ui_refs['btn_count'].label.set_color('white')
    # [FIX] Set Animated (Draw Manually)
    ui_refs['btn_count'].ax.patch.set_animated(True)
    ui_refs['btn_count'].label.set_animated(True)
    
        
    ui_refs['btn_reset'] = Button(ax_btn_reset, 'Reset', color='#662222', hovercolor='#aa0000')
    ui_refs['btn_reset'].label.set_color('white')
    # [FIX] Set Animated
    ui_refs['btn_reset'].ax.patch.set_animated(True)
    ui_refs['btn_reset'].label.set_animated(True)
    
    # Logic
    def toggle_count(_):
        global counting_active
        counting_active = not counting_active
        
        # Colors
        c_bg = '#ff6666' if counting_active else '#004400'
        c_fg = '#ffffff' if counting_active else '#ffffff'
        bg_count_rect.set_facecolor(c_bg)
        txt_count_val.set_color(c_fg)
        
        # Button State
        b = ui_refs['btn_count']
        b.label.set_text("Stop" if counting_active else "Start")
        
        # Active(Stop) -> Salmon (Red). Idle(Start) -> LightGreen
        # Direct Button Update
        if counting_active:
             b.color = '#aa0000'
             b.hovercolor = '#ff0000'
        else:
             b.color = '#004400'
             b.hovercolor = '#006600'
        
       
        # Disable Reset if counting
        br = ui_refs['btn_reset']
        if counting_active:
            br.color = '#333333'
            br.label.set_color('#666666')
            br.hovercolor = '#333333'
        else:
            br.color = '#662222'
            br.label.set_color('white')
            br.hovercolor = '#aa0000'

        # fig.canvas.draw_idle()

    def reset_count(_):
        if counting_active: return
        global ta_accum
        ta_accum = 0.0
        txt_count_val.set_text(f"TA Counter: {ta_accum:.2f}")
        # fig.canvas.draw_idle()

    ui_refs['btn_count'].on_clicked(toggle_count)
    ui_refs['btn_reset'].on_clicked(reset_count)


    # 2. Scores Panel (Center)
    r_score = [0.42, 0.015, 0.18, 0.12]
    ax_scores = reg_ax(r_score, main_view_axes)
    ax_scores.set_xticks([]); ax_scores.set_yticks([])
    ax_scores.set_facecolor('#333333')
    # Border
    rect_score_border = plt.Rectangle((0,0), 1, 1, transform=ax_scores.transAxes, fill=False, ec='#555555', lw=2, clip_on=False)
    ax_scores.add_patch(rect_score_border)

    txt_ta_score = ax_scores.text(0.5, 0.65, "INST TA: --", ha='center', va='center', fontsize=22, fontweight='bold', color='white')
    val_txt = ax_scores.text(0.5, 0.30, f"TA SET: {GSR_CENTER_VAL:.2f}", ha='center', va='center', fontsize=18, fontweight='bold', color='#aaaaaa')
    txt_ta_score.set_animated(True)
    val_txt.set_animated(True)
     
    # === STATUS BAR (Restored) ===
    ax_status = reg_ax([0.05, 0.94, 0.915, 0.04], main_view_axes)
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    ax_status.set_facecolor('#333')
    
    txt_gsr_status = ax_status.text(0.02, 0.5, "GSR: ●", color='lightgray', fontsize=11, fontweight='bold', va='center')

    # [FIX] Init with current Mic Name
    mic_str = "AUDIO: --"
    if 'audio_handler' in globals() and audio_handler.current_mic_name:
         short = audio_handler.current_mic_name
         mic_str = f"AUDIO: {short}"

    txt_audio = ax_status.text(0.67, 0.5, mic_str, color='lightgray', fontsize=11, fontweight='bold', va='center')
    
    # Rec moved left for balance (0.96 -> 0.92)
    rec_text = ax_status.text(0.92, 0.5, "● REC", color='red', fontsize=11, fontweight='bold', va='center', visible=False)
    
    # [FIX] Set Animated
    txt_gsr_status.set_animated(True)
    # txt_hrm_status.set_animated(True)
    txt_audio.set_animated(True)
    rec_text.set_animated(True)
    ui_refs['rec_text'] = rec_text
    ui_refs['txt_audio'] = txt_audio # [FIX] Register for UI Callback
    
    ui_refs['txt_gsr_status'] = txt_gsr_status

    # Record Button (Moved Left)
    # [FIX] Big Record Button (Double Height)
    r_rc = [0.05, 0.015, 0.12, 0.12]
    
    ax_rec = reg_ax(r_rc, main_view_axes)
    ax_rec.set_zorder(1000) # [FIX] Ensure top clickability
    ui_refs['btn_rec'] = Button(ax_rec, "Record", color='#004400', hovercolor='#006600')
    ui_refs['btn_rec'].label.set_color('white')
    # [FIX] Animated
    ui_refs['btn_rec'].ax.patch.set_animated(True)
    ui_refs['btn_rec'].label.set_animated(True)
    
    import tkinter as tk

    # [NEW] Globals for Auto-Start Sequence
    pending_rec = False
    pending_notes = ""
    session_start_ta = 0.0

    # ==========================================
    #           PROCESS LOGIC
    # ==========================================
    def show_process_selector(event=None):
        if not process_runner:
            log_msg("Err: Process Runner not init")
            return
            
        # Create Popup
        # [FIX] Use existing root (Toplevel defaults to default root)
        dlg = tk.Toplevel()
        dlg.title("Select Process")
        dlg.geometry("400x300")
        dlg.configure(bg='#2b2b2b')
        
        lbl = ttk.Label(dlg, text="Available Processes:", background='#2b2b2b', foreground='white', font=('Arial', 12, 'bold'))
        lbl.pack(pady=10)
        
        # Listbox
        names = process_runner.list_processes()
        lb = tk.Listbox(dlg, bg='#333', fg='white', font=('Arial', 11))
        for n in names: lb.insert(tk.END, n)
        lb.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        def on_select():
            sel = lb.curselection()
            if not sel: return
            name = lb.get(sel[0])
            dlg.destroy()
            # Start
            start_process_session(name)
            
        btn = tk.Button(dlg, text="START SESSION", command=on_select, bg='#004400', fg='white', font=('Arial', 10, 'bold'))
        btn.pack(pady=10, fill=tk.X, padx=10)
        
        # Center
        dlg.update_idletasks()
        w = dlg.winfo_width(); h = dlg.winfo_height()
        x = (dlg.winfo_screenwidth() // 2) - (w // 2)
        y = (dlg.winfo_screenheight() // 2) - (h // 2)
        dlg.geometry(f"+{x}+{y}")
        
        # Bring to front?
        dlg.transient()
        dlg.grab_set()
        dlg.wait_window() # Block until closed

    def start_process_session(name):
        # [MOD] Don't start immediately. Queue via Notes Dialog.
        
        # Verify process exists first
        p_data = process_runner.get_process_data(name)
        if not p_data:
            log_msg(f"Err: Process {name} data not found")
            return
            
        # [NEW] Check for Required Assessments
        reqs = process_runner.get_required_assessments(name)
        if reqs:
             show_assessment_selector(name, reqs)
             return
             
        global pending_assessment_selection
        pending_assessment_selection = {} # Clear previous
        show_session_notes_dialog(prefill_process=name)

    def show_assessment_selector(process_name, requirements):
         """Dialog to select items for a dynamic process."""
         dlg = tk.Toplevel()
         dlg.title(f"Configure: {process_name}")
         dlg.geometry("400x500")
         dlg.configure(bg='#2b2b2b')
         
         tk.Label(dlg, text="Select Assessment Items:", bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold')).pack(pady=10)
         
         selectors = {}
         
         for req in requirements:
              frame = tk.Frame(dlg, bg='#2b2b2b')
              frame.pack(fill=tk.X, padx=20, pady=5)

              
              # Get Options
              opts = process_runner.assessments.get(req, [])
              
              # Multiselect or Single? User asked for "Choice of item". Assumed Single for now, or Listbox for multiple?
              # "track or offer the user the choice of the assessment list item" 
              # Let's support selecting ONE specific item to start with.
              
              cb = ttk.Combobox(frame, values=opts, state="readonly")
              if opts: cb.current(0)
              cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
              selectors[req] = cb
              
         def on_confirm():
              selection = {}
              for r, cb in selectors.items():
                   selection[r] = cb.get()
              
              global pending_assessment_selection
              pending_assessment_selection = selection
              dlg.destroy()
              
              # Proceed
              show_session_notes_dialog(prefill_process=process_name)
         
         tk.Button(dlg, text="CONTINUE", command=on_confirm, bg='#005500', fg='white', font=('Arial', 10, 'bold')).pack(pady=20)
         
         # Center
         dlg.update_idletasks()
         x = (dlg.winfo_screenwidth() // 2) - (dlg.winfo_width() // 2)
         y = (dlg.winfo_screenheight() // 2) - (dlg.winfo_height() // 2)
         dlg.geometry(f"+{x}+{y}")
         dlg.transient(); dlg.grab_set()

    def trigger_closing_sequence():
        """Interrupts current process and jumps to closing questions."""
        global process_in_closing_phase, process_step_idx, process_waiting_for_input, process_ending_phase
        global process_waiting_for_calib
        
        # Only trigger if we are inside a process (or running)
        # If already closing, do nothing? Or restart closing?
        if process_ending_phase: return # Already done
        
        print("[Process] Triggering Closing Sequence...", flush=True)
        log_msg("Initiating Closing Sequence...")
        
        process_in_closing_phase = True
        process_ending_phase = False # Reset this if we were at the end
        process_waiting_for_calib = False # Force out of calib wait
        
        # Reset index to 0 of closing list
        process_step_idx = 0
        
        # [NEW] Deduplicate Closing: If first closing question matches the one we just asked, skip it
        closing_questions_list = process_runner.get_closing_questions()
        if closing_questions_list and current_q_text:
             first_closing_text = closing_questions_list[0].get('text')
             if current_q_text == first_closing_text:
                  print(f"[Process] Skipping redundant closing question: {first_closing_text}")
                  process_step_idx = 1
        
        # Start immediately
        advance_process_step()

    # Removed duplicate def advance_process_step
        
    def advance_process_step():
        global process_step_idx, process_waiting_for_input, active_event_label
        global active_process_data, process_in_closing_phase
        global current_q_set, current_q_id, current_q_text # [NEW]
        
        # Decide source of steps
        if process_in_closing_phase:
             steps = process_runner.get_closing_questions()
             prefix = "CLOSING"
        else:
             if not active_process_data: return
             steps = active_process_data.get('steps', [])
             prefix = "STEP"
        
        if process_step_idx >= len(steps):
             # [MOD] Auto-trigger Closing Questions if not already in closing phase
             if not process_in_closing_phase and not process_ending_phase:
                  trigger_closing_sequence()
                  return
             
             # If we simply ran out of steps AND we are in closing phase (or closing valid), End Session
             enter_process_ending_phase()
             return
            
        # Get Step
        step = steps[process_step_idx]
        text = step.get('text', "")
        audio_file = step.get('audio_file', "")
        
        # [NEW] Extract Metadata
        current_q_set = step.get('set', "")
        current_q_id = step.get('question', "")
        current_q_text = text
        
        # Visuals
        log_msg(f"[{prefix}] {process_step_idx+1}: {text}")
        if 'txt_process_overlay' in ui_refs:
            ui_refs['txt_process_overlay'].set_text(text)
            ui_refs['txt_process_overlay'].set_color('#FFCC00') # Orange/Gold
            ui_refs['txt_process_overlay'].set_fontsize(13) # [MOD] Smaller instruction
            ui_refs['txt_process_overlay'].set_visible(True)
            
        # Audio
        threading.Thread(target=lambda: play_step_audio(text, audio_file), daemon=True).start()
        
        # [NEW] Auto-Advance Check
        auto_adv = step.get('auto_advance', False)
        
        process_waiting_for_input = not auto_adv # If auto-adv, don't wait for input
        active_event_label = f"{prefix}_{process_step_idx+1}_START"
        
        # Prepare for next
        process_step_idx += 1
        
        if auto_adv:
            # Threaded wait for audio then advance
            def _auto_adv_task():
                # Give TTS a moment to start
                time.sleep(1.0)
                while process_runner.is_playing():
                    time.sleep(0.1)
                time.sleep(0.5) # Short pause after audio
                if active_process_name: # Still running?
                    advance_process_step()
            threading.Thread(target=_auto_adv_task, daemon=True).start()

    def trigger_closing_sequence():
        """Interrupts current process and jumps to closing questions."""
        global process_in_closing_phase, process_step_idx, process_waiting_for_input, process_ending_phase
        
        # Only trigger if we are inside a process (or running)
        # If already closing, do nothing? Or restart closing?
        if process_ending_phase: return # Already done
        
        print("[Process] Triggering Closing Sequence...", flush=True)
        log_msg("Initiating Closing Sequence...")
        
        process_in_closing_phase = True
        process_ending_phase = False # Reset this if we were at the end
        process_waiting_for_calib = False # Force out of calib wait
        
        # Reset index to 0 of closing list
        process_step_idx = 0
        
        # Start immediately
        advance_process_step()

    def enter_process_ending_phase():
        """Shows session complete message and waits for final confirmation."""
        global process_waiting_for_input, process_ending_phase
        
        log_msg("[Process] All steps complete. Waiting for user confirmation.")
        
        if 'txt_process_overlay' in ui_refs:
            ui_refs['txt_process_overlay'].set_text("SESSION COMPLETE\n(Press SPACE or ENTER to Finish)")
            ui_refs['txt_process_overlay'].set_color('#00FF00') # Green
            ui_refs['txt_process_overlay'].set_visible(True)
            
        process_ending_phase = True
        process_waiting_for_input = True # Enable key handler to catch the final press

    def reset_process_state():
        global active_process_name, active_process_data, process_step_idx
        global process_waiting_for_calib, process_waiting_for_input
        global process_ending_phase, process_in_closing_phase
        global current_q_set, current_q_id, current_q_text
        
        print("[Process] Resetting all process state variables.")
        active_process_name = None
        active_process_data = None
        process_step_idx = -1
        process_waiting_for_calib = False
        process_waiting_for_input = False
        process_ending_phase = False
        process_in_closing_phase = False
        current_q_set = ""
        current_q_id = ""
        current_q_text = ""

    def speak_confirmation(text):
        """Plays a brief voice confirmation for settings changes."""
        if process_runner and process_runner.audio_enabled:
            def _task():
                # Locate or generate audio
                f_path = process_runner.prepare_step_audio(text, None)
                if f_path:
                    # Optional: wait for previous audio to finish?
                    # For confirmation, we just want it to be responsive.
                    try:
                        while process_runner.is_playing():
                            time.sleep(0.1)
                        process_runner.play_audio_file(f_path)
                    except: pass
            threading.Thread(target=_task, daemon=True).start()

    def play_step_audio(text, audio_file):
        # Prepare (can take time for TTS)
        f_path = process_runner.prepare_step_audio(text, audio_file)
        if f_path:
            # wait for channel free?
            while process_runner.is_playing():
                time.sleep(0.1)
            process_runner.play_audio_file(f_path)

    def finish_process_session():
        global active_process_name, active_process_data, active_process_input
        log_msg("Process Session Complete.")
        
        if 'txt_process_overlay' in ui_refs:
            ui_refs['txt_process_overlay'].set_text("SESSION COMPLETE")
            ui_refs['txt_process_overlay'].set_color('white')
            
        # Reset Logic
        reset_process_state()
        
        # Stop Recording? User requested step 9: "stop recording"
        # Confirm "End of Session" -> then Stop?
        # Flow: Last question finished -> confirm -> End msg -> Stop.
        # Implemented: End msg is shown.
        # Trigger stop?
        if is_recording:
            toggle_rec(None)

    def start_actual_recording():
        global is_recording, f_gsr, writer_gsr, recording_start_time_obj
        global notes_filename, audio_filename # [FIX] Restored audio_filename
        # [FIX] Audio Globals Removed (Using AudioHandler)
        global pending_rec, pending_notes, session_start_ta, counting_active, ta_accum # [FIX] Globals
        global ZOOM_COEFFICIENT # [NEW]
        
        # [REQ] Auto-reset Zoom on Session Start
        ZOOM_COEFFICIENT = 1.0
        update_zoom_ui()
        
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
             # fname_hrm = os.path.join(SESSION_DIR, "HRM.csv") # [NEW]
             audio_filename = os.path.join(SESSION_DIR, "Audio.wav") # [FIX] Define Audio Path

             notes_filename = os.path.join(SESSION_DIR, "notes.txt")
             
             with open(notes_filename, "w") as f:
                 f.write(f"Session Notes - {ts_str}\n")
                 f.write("-" * 30 + "\n")
                 f.write(pending_notes if pending_notes else "No notes provided.")
             
             # Initialize GSR CSV
             f_gsr = open(fname_gsr, 'w', newline='')
             writer_gsr = csv.writer(f_gsr)
             writer_gsr.writerow(["Timestamp", "Elapsed", "TA", "TA Counter", "TA SET", "Sensitivity", "Window_Size", "Motion", "Pivot", "Notes", "Pattern", "Log_TA", "Question_Set", "Question_ID", "Question_Text"])
             
             # [NEW] Record Session Start Note in CSV
             ts_now = datetime.now().strftime('%H:%M:%S.%f')
             win = get_effective_window()
             log_ta = math.log10(max(0.01, session_start_ta))
             writer_gsr.writerow([ts_now, "00:00:00.00000", f"{session_start_ta:.5f}", "0.00000", f"{GSR_CENTER_VAL:.3f}", f"{1.0/win:.3f}", f"{win:.3f}", 0, f"{CALIB_PIVOT_TA:.3f}", "RECORDING_STARTED", "", f"{log_ta:.5f}", "", "", ""])
             
             recording_start_time_obj = datetime.now()
             is_recording = True 
             
             audio_handler.start_recording(audio_filename)
             audio_handler.sync_audio_stream(current_view)
             
             ui_refs['btn_rec'].label.set_text("Stop")
             ui_refs['btn_rec'].color = '#aa0000'
             ui_refs['btn_rec'].hovercolor = '#ff0000'
             rec_text.set_visible(True)
             
             # Set Static Info
             s_date = recording_start_time_obj.strftime("%d %B %Y")
             s_time = recording_start_time_obj.strftime("%H:%M")
             if 'txt_sess_date' in ui_refs: ui_refs['txt_sess_date'].set_text(f"Date: {s_date}")
             if 'txt_sess_time' in ui_refs: ui_refs['txt_sess_time'].set_text(f"Time: {s_time}")
             
             log_msg(f"Started: {ts_str}")
             
        except Exception as ex: 
            log_msg(f"Start Err: {ex}")
            is_recording = False

    # [NEW] Refactored Notes Dialog
    def show_session_notes_dialog(prefill_process=None, manual_trigger=False):
        global pending_notes, pending_rec, calib_mode, is_recording, current_user_name
        global pending_assessment_selection # [FIX] Moved to top
        
        # [NEW] Check Mic Selection First
        if audio_handler.selected_device_idx is None:
             log_msg("Err: No Mic Selected!")
             return

        # [FIX] Use existing root
        dlg = tk.Toplevel()
        dlg.title("Session Notes")
        dlg.geometry("500x400")
        dlg.configure(bg='#f0f0f0')
        
        tk.Label(dlg, text="Session Details", font=("Arial", 12, "bold"), bg='#f0f0f0').pack(pady=10)
        
        # Prefill Logic
        p_user = current_user_name
        p_proc = prefill_process if prefill_process else ""
        
        # [NEW] Append Selection to Notes
        if pending_assessment_selection:
             sel_str = ", ".join(pending_assessment_selection.values())
             p_proc += f" [{sel_str}]"
        
        template_text = f"Client Name: {p_user}\n\nProcess Run: {p_proc}\n\nOther Notes:"
        
        txt = tk.Text(dlg, width=50, height=13, font=("Arial", 10))
        txt.pack(padx=10, pady=5, expand=True, fill='both')
        txt.insert("1.0", template_text)
        txt.focus_set()
        
        # [NEW] Audio Toggle for Session
        audio_var = tk.BooleanVar(value=process_runner.audio_enabled if process_runner else True)
        chk_audio = tk.Checkbutton(dlg, text="Play Question Audio", variable=audio_var, bg='#f0f0f0')
        chk_audio.pack(pady=5)
        
        user_choice = [None] # [None] or "start"

        def on_submit():
            user_choice[0] = "start"
            # [FIX] Get text BEFORE destroy
            global pending_notes
            pending_notes = txt.get("1.0", "end-1c")
            
            # [NEW] Apply Audio Setting
            if process_runner:
                process_runner.audio_enabled = audio_var.get()
                
            dlg.destroy()
            
        def on_cancel():
            dlg.destroy()
            
        btn_frame = tk.Frame(dlg, bg='#f0f0f0')
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Start Session", command=on_submit, bg="#ddffdd", height=2, width=15).pack(side='left', padx=10)
        tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side='left', padx=10)
        
        # Center
        dlg.update_idletasks()
        x = (dlg.winfo_screenwidth() // 2) - (dlg.winfo_width() // 2)
        y = (dlg.winfo_screenheight() // 2) - (dlg.winfo_height() // 2)
        dlg.geometry(f"+{x}+{y}")
        
        dlg.transient()
        dlg.grab_set()
        dlg.wait_window()
        
        if user_choice[0] == "start":
             # START LOGIC
             start_actual_recording()
             pending_rec = True
             if prefill_process:
                   # PROCESS START
                   reset_process_state()
                   global active_process_name, active_process_data, process_step_idx, process_waiting_for_calib, process_waiting_for_input
                   # Verify we have the data (should be cached)
                   if process_runner:
                       # [NEW] Check for Pending Selection

                       if pending_assessment_selection:
                            print(f"[Process] Compiling dynamic steps with: {pending_assessment_selection}")
                            custom_steps = process_runner.compile_process_dynamic(prefill_process, pending_assessment_selection)
                            if process_runner.starting_questions:
                                 # [FIX] Deduplicate SOS from Prefix if redundant
                                 prefix_steps = process_runner.starting_questions
                                 if custom_steps and prefix_steps:
                                      if custom_steps[0].get('text') == prefix_steps[-1].get('text'):
                                           prefix_steps = prefix_steps[:-1]
                                 custom_steps = prefix_steps + custom_steps
                                 
                            # Create temporary process data
                            active_process_name = prefill_process
                            active_process_data = {
                                "name": prefill_process,
                                "steps": custom_steps
                            }
                            if 'txt_process_overlay' in ui_refs:
                                display_item = next(iter(pending_assessment_selection.values())) if pending_assessment_selection else "Dynamic"
                                ui_refs['txt_process_overlay'].set_text(f"PROCESS: {prefill_process} [{display_item}]\n(Waiting for Calibration)")
                                ui_refs['txt_process_overlay'].set_visible(True)

                            # Clear after use
                            pending_assessment_selection = {}
                            
                            # [FIX] Initialize State
                            process_step_idx = 0
                            process_waiting_for_calib = True
                            process_waiting_for_input = False
                       else:
                            p_data = process_runner.get_process_data(prefill_process)
                            if p_data:
                                 print(f"[Process] Starting Standard: {prefill_process} (Steps: {len(p_data['steps'])})")
                                 active_process_name = prefill_process
                                 # [NEW] Prepend Starting Questions (Create Copy to avoid mutation)
                                 if process_runner.starting_questions:
                                      print(f"[Process] Prepending {len(process_runner.starting_questions)} Starting Questions")
                                      # [FIX] Deduplicate SOS from Prefix if redundant
                                      prefix_steps = process_runner.starting_questions
                                      
                                      # Deduplicate: if first step of process is same as last step of prefix
                                      if p_data['steps'] and prefix_steps:
                                           first_p_text = p_data['steps'][0].get('text') if isinstance(p_data['steps'][0], dict) else process_runner.get_question_text(p_data['steps'][0])
                                           last_pre_text = prefix_steps[-1].get('text')
                                           if first_p_text == last_pre_text:
                                                prefix_steps = prefix_steps[:-1]

                                      # Shallow copy dict, new list for steps
                                      p_copy = p_data.copy()
                                      p_copy['steps'] = prefix_steps + p_data['steps']
                                      active_process_data = p_copy
                                 else:
                                      active_process_data = p_data
                            process_step_idx = 0
                            process_waiting_for_calib = True
                            process_waiting_for_input = False
                            
                            if 'txt_process_overlay' in ui_refs:
                                 ui_refs['txt_process_overlay'].set_text(f"PROCESS: {prefill_process}\n(Waiting for Calibration)")
                                 ui_refs['txt_process_overlay'].set_visible(True)
             
             # ALWAYS start Calibration
             start_calibration(None)
             
        elif manual_trigger:
             # If cancelled during manual toggle, ensure state is reset
             pass

    def toggle_rec(_):
        #print("DEBUG: toggle_rec clicked") # [DEBUG]
        global is_recording, f_gsr, writer_gsr, recording_start_time_obj
        global notes_filename, audio_filename
        global pending_rec, pending_notes, calib_mode, calib_step, calib_phase, calib_start_time, calib_base_ta, calib_min_ta, counting_active
        
        if not is_recording:
             # MANUAL START -> Notes Dialog
             show_session_notes_dialog(prefill_process=None, manual_trigger=True)
             
        else:
             # STOP Logic (Same as before)
             is_recording = False
             audio_handler.is_recording = False
             if counting_active: toggle_count(None) # [REQ] Stop TA Counter on Session Stop
             
             # If stopping during calibration?
             if calib_mode:
                 calib_mode = False
                 pending_rec = False
                 log_msg("Start Aborted.")
                 
                 # [FIX] Must Reset UI even if aborting early
                 ui_refs['btn_rec'].label.set_text("Record")
                 ui_refs['btn_rec'].color = '#004400'
                 ui_refs['btn_rec'].hovercolor = '#006600'
                 try: 
                     # fig.canvas.draw_idle() 
                     pass
                 except Exception: pass
                 return # Exit early, no session file to close?
              
             ui_refs['btn_rec'].label.set_text("Record")
             ui_refs['btn_rec'].color = '#004400'
             ui_refs['btn_rec'].hovercolor = '#006600'
             try: 
                 # fig.canvas.draw_idle() 
                 pass
             except Exception: pass # [FIX] Non-blocking refresh 
                          
             # [NEW] Record Session End in CSV
             try:
                 if writer_gsr:
                     ts_now = datetime.now().strftime('%H:%M:%S.%f')
                     elapsed_str = "00:00:00.00000"
                     if recording_start_time_obj:
                         elapsed_val = (datetime.now() - recording_start_time_obj).total_seconds()
                         elapsed_str = format_elapsed(elapsed_val)
                     win = get_effective_window()
                     log_ta = math.log10(max(0.01, latest_gsr_ta))
                     writer_gsr.writerow([ts_now, elapsed_str, f"{latest_gsr_ta:.5f}", f"{ta_accum:.5f}", f"{GSR_CENTER_VAL:.3f}", f"{1.0/win:.3f}", f"{win:.3f}", 0, f"{CALIB_PIVOT_TA:.3f}", "RECORDING_ENDED", "", f"{log_ta:.5f}", "", "", ""])
                     f_gsr.flush()
             except Exception: pass

             audio_handler.stop_recording()
             audio_handler.sync_audio_stream(current_view) 
             
             # [NEW] Append Summary to Notes
             try:
                 if recording_start_time_obj and notes_filename:
                     end_time = datetime.now()
                     dur = end_time - recording_start_time_obj
                     total_sec = int(dur.total_seconds())
                     hours, remainder = divmod(total_sec, 3600)
                     mins, secs = divmod(remainder, 60)
                     final_len_str = f"{hours:02}:{mins:02}:{secs:02}"
                     
                     start_fmt = recording_start_time_obj.strftime("%H:%M:%S")
                     
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
    r_detail = [0.62, 0.015, 0.20, 0.12]
    ax_detail = reg_ax(r_detail, main_view_axes)
    ax_detail.set_xticks([]); ax_detail.set_yticks([])
    ax_detail.set_facecolor('#333333')
    rect_det_border = plt.Rectangle((0,0), 1, 1, transform=ax_detail.transAxes, fill=False, ec='#555555', lw=2, clip_on=False)
    ax_detail.add_patch(rect_det_border)
    
    # Title
    ax_detail.text(0.5, 0.85, "Session Detail", ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Text Fields
    # [NEW] User Name
    # Date/Time shifted down (0.60 -> 0.55 etc)
    txt_sess_user = ax_detail.text(0.05, 0.68, f"User: {current_user_name}", ha='left', va='center', fontsize=10, fontweight='bold', color='#FFCC00')
    txt_sess_date = ax_detail.text(0.05, 0.48, "Date: --", ha='left', va='center', fontsize=9, color='#ccc')
    txt_sess_time = ax_detail.text(0.05, 0.32, "Time: --", ha='left', va='center', fontsize=9, color='#ccc')
    txt_sess_len  = ax_detail.text(0.05, 0.15, "Duration : 00:00:00", ha='left', va='center', fontsize=10, fontweight='bold', color='white')
    
    txt_sess_user.set_animated(True)
    txt_sess_date.set_animated(True)
    txt_sess_time.set_animated(True)
    txt_sess_len.set_animated(True)
    
    ui_refs['txt_sess_user'] = txt_sess_user
    ui_refs['txt_sess_date'] = txt_sess_date
    ui_refs['txt_sess_time'] = txt_sess_time
    ui_refs['txt_sess_len'] = txt_sess_len

    # --- SYSTEM CONTROLS PANEL ---
    # [FIX] Extended Height to fit 5 buttons
    r_sys_bg = [0.835, 0.355, 0.13, 0.30]
    ax_system_bg = reg_ax(r_sys_bg, main_view_axes)
    ax_system_bg.set_facecolor('#333333')
    ax_system_bg.set_xticks([]); ax_system_bg.set_yticks([])
    # Border
    rect_sys_border = plt.Rectangle((0,0), 1, 1, transform=ax_system_bg.transAxes, fill=False, ec='#555555', lw=2, clip_on=False)
    ax_system_bg.add_patch(rect_sys_border)
    # Title
    ax_system_bg.text(0.5, 0.93, "SYSTEM CONTROLS", transform=ax_system_bg.transAxes, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ui_refs['ax_system_bg'] = ax_system_bg

    # [FIX] Relocated Buttons (Now inside SYSTEM CONTROLS)
    r_calib = [0.85, 0.575, 0.10, 0.04]
    ax_calib = reg_ax(r_calib, main_view_axes)
    ui_refs['btn_calib'] = Button(ax_calib, "Calibrate", color='#004488', hovercolor='#0066aa')
    ui_refs['btn_calib'].label.set_color('white') #'#eee'
    ui_refs['btn_calib'].ax.patch.set_animated(True)
    ui_refs['btn_calib'].label.set_animated(True)    
    ui_refs['btn_calib'].on_clicked(start_calibration)
    
    ui_refs['btn_calib'].on_clicked(start_calibration)
    
    # [NEW] Processes Button
    r_proc = [0.85, 0.525, 0.10, 0.04]
    ax_proc = reg_ax(r_proc, main_view_axes)
    ui_refs['btn_processes'] = Button(ax_proc, "Processes", color='#550055', hovercolor='#770077')
    ui_refs['btn_processes'].label.set_color('white')
    ui_refs['btn_processes'].ax.patch.set_animated(True)
    ui_refs['btn_processes'].label.set_animated(True)
    ui_refs['btn_processes'].on_clicked(show_process_selector)

    # [NEW] Manual Button Standardized (Shifted Down)
    r_man = [0.85, 0.475, 0.10, 0.04]
    ax_man = reg_ax(r_man, main_view_axes)
    ui_refs['btn_manual'] = Button(ax_man, "Manual", color='#2980b9', hovercolor='#3498db')
    ui_refs['btn_manual'].label.set_color('white')
    ui_refs['btn_manual'].ax.patch.set_animated(True)
    ui_refs['btn_manual'].label.set_animated(True)
    ui_refs['btn_manual'].on_clicked(show_manual_popup)

    # [NEW] Viewer Button (Below Manual - Shifted Down)
    r_view = [0.85, 0.425, 0.10, 0.04]
    ax_view = reg_ax(r_view, main_view_axes)
    ui_refs['btn_viewer'] = Button(ax_view, "Viewer", color='#552255', hovercolor='#773377')
    ui_refs['btn_viewer'].label.set_color('white')
    ui_refs['btn_viewer'].ax.patch.set_animated(True)
    ui_refs['btn_viewer'].label.set_animated(True)

    # [FIX] Relocated Buttons (Now inside SYSTEM CONTROLS)
    r_ts = [0.85, 0.375, 0.10, 0.04] # [FIX] Shifted down to avoid overlap
    ax_to_set = reg_ax(r_ts, main_view_axes)
    ui_refs['btn_to_settings'] = Button(ax_to_set, "Settings >", color='#444444', hovercolor='#666666')
    ui_refs['btn_to_settings'].label.set_color('white')
    ui_refs['btn_to_settings'].ax.patch.set_animated(True)
    ui_refs['btn_to_settings'].label.set_animated(True)
    # [NEW] Exit Button (Bottom Right, Aligned with column)
    r_exit = [0.85, 0.025, 0.10, 0.04] 
    ax_exit = reg_ax(r_exit, main_view_axes)
    ax_exit.set_zorder(100)
    ui_refs['btn_exit'] = Button(ax_exit, "EXIT", color='#A00', hovercolor='#F00')
    ui_refs['btn_exit'].label.set_color('white')
    ui_refs['btn_exit'].label.set_fontsize(11) 
    ui_refs['btn_exit'].label.set_fontweight('bold')
    ui_refs['btn_exit'].ax.patch.set_animated(True)
    ui_refs['btn_exit'].label.set_animated(True)
    ui_refs['btn_exit'].on_clicked(lambda e: os._exit(0)) # Force Hard Exit

    # VIEW LOGIC
    viewer_frame = None
    sess_viewer = None
    
    def enter_viewer(event):
        global current_view, viewer_frame, sess_viewer, timer
        
        # 1. Stop Live Updates
        timer.stop()
        current_view = 'viewer'
        
        # 2. Release Audio Input
        log_msg("Switching to Viewer... (Mic Paused)")
        try:
             # Stop recording if active?
             if is_recording:
                  toggle_rec(None) # Safe stop
             
             # Pause Input Stream
             if audio_handler.audio_stream:
                 try: audio_handler.audio_stream.stop()
                 except: pass
        except Exception as e: print(f"Audio Stop Err: {e}")

        # 3. Hide Canvas
        canvas_widget = fig.canvas.get_tk_widget()
        canvas_widget.pack_forget()
        
        # 4. Show Viewer Frame
        root_window = fig.canvas.manager.window
        
        if viewer_frame is None:
             viewer_frame = tk.Frame(root_window, bg='#222')
             # Create Viewer
             sess_viewer = SessionViewer(viewer_frame, audio_handler, on_close_callback=exit_viewer)
        
        viewer_frame.pack(fill=tk.BOTH, expand=True)
        
    def exit_viewer():
        global current_view, viewer_frame, sess_viewer
        
        # 0. Stop playback and cleanup viewer state directly
        if sess_viewer:
             try: 
                 sess_viewer.is_playing = False
                 sess_viewer.stop_playback(reset=True)
                 if sess_viewer.timer_id:
                     sess_viewer.master.after_cancel(sess_viewer.timer_id)
                     sess_viewer.timer_id = None
             except: pass

        # 1. Hide Viewer
        if viewer_frame:
             try: viewer_frame.pack_forget()
             except: pass
             
        # 2. Show Canvas
        canvas_widget = fig.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 3. Stop any playback audio and resume live audio
        log_msg("Resuming Live Mode...")
        try:
             # Stop playback first
             audio_handler.stop_playback()
             # Then ensure stream is synced/started for main view
             audio_handler.sync_audio_stream('main') 
        except Exception as e: print(f"Audio Resume Err: {e}")
        
        # 4. Resume Timer
        current_view = 'main'
        # Force redraw to clear artifacts
        fig.canvas.draw()
        try: timer.start()
        except: pass

    ui_refs['btn_viewer'].on_clicked(enter_viewer)

    # --- SETTINGS PAGE ELEMENTS ---
    # [FIX] Full Screen Blind to cover Main Page artifacts during Blitting
    ax_blind = reg_ax([0,0,1,1], settings_view_axes)
    ax_blind.set_xticks([]); ax_blind.set_yticks([])
    ax_blind.set_facecolor('#2b2b2b') 
    ax_blind.set_zorder(-5) 
    
    rect_audio = [0.35, 0.20, 0.28, 0.32] 
    ax_audio_bg = create_panel_ax(rect_audio, "Audio Input Control")
    ui_refs['ax_audio_bg'] = ax_audio_bg
    
    # 3. Audio Input Control
    ax_mic_lbl = reg_ax([rect_audio[0]+0.02, rect_audio[1]+0.22, rect_audio[2]-0.04, 0.06], settings_view_axes)
    ax_mic_lbl.axis('off')
    ui_refs['text_mic_name'] = ax_mic_lbl.text(0, 0.5, "NO MIC", va="center", ha="left", fontsize=9, color='white')
    
    r_meter = [rect_audio[0]+0.02, rect_audio[1]+0.16, rect_audio[2]-0.09, 0.02] 
    ax_level = reg_ax(r_meter, settings_view_axes)
    ax_level.set_xlim(0, 1.0); ax_level.set_ylim(-0.5, 0.5) 
    ax_level.set_xticks([]); ax_level.set_yticks([])
    ax_level.set_facecolor("#333")
    ax_level.set_zorder(10) 
    bar_level = ax_level.barh([0], [0], color='green', height=1.0)
    
    ax_lvl_txt = reg_ax([rect_audio[0]+rect_audio[2]-0.06, rect_audio[1]+0.15, 0.05, 0.04], settings_view_axes)
    ax_lvl_txt.axis('off')
    ui_refs['text_level'] = ax_lvl_txt.text(0.5, 0.5, "0%", ha='center', va='center', fontsize=8, color='cyan')

    r_msel = [rect_audio[0]+0.02, rect_audio[1]+0.10, 0.20, 0.035]
    ax_msel = reg_ax(r_msel, settings_view_axes)
    ax_msel.set_zorder(20) # [FIX] Ensure visible
    ui_refs['btn_select_mic'] = Button(ax_msel, 'Select Input...', color='#666666', hovercolor='#888888')
    ui_refs['btn_select_mic'].label.set_color('white')
    ui_refs['btn_select_mic'].on_clicked(lambda e: audio_handler.open_audio_select())

    # --- VOICE & AUDIO PANEL ---
    rect_voice = [0.05, 0.15, 0.28, 0.38] # [FIX] Taller panel
    ax_voice_bg = create_panel_ax(rect_voice, "Voice & Audio")
    ui_refs['ax_voice_bg'] = ax_voice_bg
    
    # Accent Selection
    ax_voice_bg.text(0.05, 0.90, "Voice Accent:", transform=ax_voice_bg.transAxes, color='white', fontsize=10) # Higher Y
    
    accents = [
        ("UK", "co.uk"), ("US", "com"), ("AU", "com.au"), ("IN", "co.in"),
        ("CA", "ca"), ("IE", "ie"), ("NZ", "nz"), ("ZA", "za")
    ]
    accent_btns = {}
    
    def set_accent(tld):
        if process_runner:
            process_runner.voice_tld = tld
            # Update colors
            for b_tld, btn in accent_btns.items():
                btn.color = '#444444' if b_tld != tld else '#0066aa'
            save_config() # [NEW] Immediate Save
            
            # [NEW] Descriptive Voice Confirmation
            name_map = {"co.uk": "UK", "com": "US", "com.au": "Australia", "co.in": "India", "ca": "Canada", "ie": "Ireland", "nz": "New Zealand", "za": "South Africa"}
            disp_name = name_map.get(tld, tld)
            speak_confirmation(f"Voice set to {disp_name} style.")
            fig.canvas.draw_idle()
    
    for i, (label, tld) in enumerate(accents):
        # Two rows of 4
        row = i // 4
        col = i % 4
        # Increased spacing: rect_voice[1] + 0.26 down to 0.19
        r_acc = [rect_voice[0] + 0.02 + (col * 0.065), rect_voice[1] + 0.26 - (row * 0.045), 0.06, 0.035]
        ax_acc = reg_ax(r_acc, settings_view_axes)
        btn = Button(ax_acc, label, color='#444444', hovercolor='#666666')
        btn.label.set_color('white')
        btn.label.set_fontsize(8) # [FIX] Prevent Clipping
        btn.on_clicked(lambda e, t=tld: set_accent(t))
        accent_btns[tld] = btn
    
    ui_refs['accent_btns'] = accent_btns
    
    # Gender Selection [NEW]
    ax_voice_bg.text(0.05, 0.42, "Voice Gender:", transform=ax_voice_bg.transAxes, color='white', fontsize=10) # Adjusted Y
    
    gender_btns = {}
    def set_gender(gen):
        if process_runner:
            process_runner.voice_gender = gen
            for g, btn in gender_btns.items():
                btn.color = '#444444' if g != gen else '#006688'
            save_config() # [NEW] Immediate Save
            speak_confirmation(f"Voice Gender changed to {gen}.")
            fig.canvas.draw_idle()

    for i, gen in enumerate(["Male", "Female"]):
        r_gen = [rect_voice[0] + 0.02 + (i * 0.13), rect_voice[1] + 0.07, 0.12, 0.035] # Lower Y
        ax_gen = reg_ax(r_gen, settings_view_axes)
        btn = Button(ax_gen, gen, color='#444444', hovercolor='#666666')
        btn.label.set_color('white')
        btn.on_clicked(lambda e, g=gen: set_gender(g))
        gender_btns[gen] = btn
    
    ui_refs['gender_btns'] = gender_btns
    
    # Audio Toggle Removed as per User Request (Selected at process run time)


    r_gain = [rect_audio[0] + 0.02, rect_audio[1] + 0.01, rect_audio[2] - 0.04, 0.03]
    ax_gain = reg_ax(r_gain, settings_view_axes)
    ui_refs['slide_gain'] = Slider(ax_gain, 'Mic Gain', 1.0, 10.0, valinit=audio_handler.current_mic_gain, color='lime')
    ui_refs['slide_gain'].on_changed(lambda v: setattr(audio_handler, 'current_mic_gain', v))


    
    # Back Button
    r_bk = [0.05, 0.90, 0.10, 0.04]
    ax_back = reg_ax(r_bk, settings_view_axes)
    ui_refs['btn_back'] = Button(ax_back, "< Back", color='#444444', hovercolor='#666')
    ui_refs['btn_back'].label.set_color('white')
    # [FIX] Do NOT animate button back, as Settings View uses standard draw()
    # ui_refs['btn_back'].ax.patch.set_animated(True)
    # ui_refs['btn_back'].label.set_animated(True)
    
    # [SYSTEM LINE REMOVED]
    
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
    def handle_calibration_phase(step_idx, phase, start_time, base_ta, min_ta, curr_ta, elapsed, is_steady, step_start_time=0.0):
        # Universal Phase Handler
        # Returns: (new_phase, new_min_ta, event_label, msg, new_base_ta, new_calib_start_time)
        
        new_phase = phase
        new_min_ta = min_ta
        new_base_ta = base_ta
        new_start_time = start_time
        evt = None
        txt = ""
        
        is_breath = (step_idx == 4)
        prefix = f"CALIB {step_idx}/4"
        
        if phase == 0: # WAITING FOR DROP/BREATH
             is_large_drop = (base_ta - curr_ta) > 0.05
             
             if not is_breath:
                  # SQUEEZE (Steps 1-3)
                  # [FIX] Logic: Wait for (2s AND Steady) OR large drop (already squeezed)
                  if elapsed < 2.0 or (not is_steady and not is_large_drop):
                       txt = f"{prefix}: RELAX HAND..."
                       # Hysteresis (0.002)
                       new_base_ta = update_hysteresis(curr_ta, base_ta, 0.002)
                  else:
                       txt = f"{prefix}: SQUEEZE SENSOR\n(WAITING FOR DROP)"
                       if curr_ta < base_ta - 0.05:
                            new_phase = 1
                            new_min_ta = curr_ta
                            new_start_time = time.time()
                            evt = f"SQUEEZE_{step_idx}_DROP"
             else:
                  # BREATH (Step 4)
                  # Timeout handled externally or here? Externally is easier for "return" logic
                  if elapsed < 2.0:
                       txt = f"{prefix}: RELAX HAND..."
                       new_base_ta = update_hysteresis(curr_ta, base_ta, 0.05) # [FIX] Typos in original: 0.05
                  else:
                       txt = f"{prefix}: DEEP BREATH IN\n(WAITING FOR DROP)"
                       if curr_ta < base_ta - 0.02:
                            new_phase = 1
                            new_min_ta = curr_ta
                            new_start_time = time.time()
                            evt = "BREATH_DROP"
                            
        elif phase == 1: # TRACKING DROP
             txt = f"{prefix}: AND RELEASE..."
             if curr_ta < min_ta:
                  new_min_ta = curr_ta
                  if is_breath: new_start_time = time.time() # Reset stability timer for Breath Only?
             
             # Detect Release / Stability
             # Squeeze: Release > 75% or +0.05
             # Breath: Stabilize > 1.5s
             
             if not is_breath:
                  # SQUEEZE RECOVERY LOGIC
                  current_drop = base_ta - min_ta
                  recovery_target = min_ta + (current_drop * 0.75) if current_drop > 0.05 else min_ta + 0.05
                  if curr_ta > recovery_target:
                       new_phase = 3
                       new_start_time = time.time()
                       evt = f"SQUEEZE_{step_idx}_RELEASE"
             else:
                  # BREATH STABILITY LOGIC
                  # Need external time_in_phase? Passed as elapsed?
                  # No, elapsed is phase duration.
                  # Step duration is diff.
                  # Logic inside update_loop is tricky for Breath Phase 1.
                  # Let's keep Phase 1 Breath simple here:
                  # Return special flag or handle logic.
                  # Actually, checking stability in helper is fine if we pass params.
                  pass # Handle in loop for specific breath logic complexity

        elif phase == 3: # STABILIZATION
             txt = "CALIB: STABILIZING..."
        
        return (new_phase, new_min_ta, evt, txt, new_base_ta, new_start_time)
        
    def update_hysteresis(curr, base, drop_limit):
        # 1. Follow Rises (Recovery)
        if curr > base: return curr
        # 2. Follow TINY Drops (Micro-Drift) only.
        if (base - curr) < drop_limit: return curr
        # Else: Hold Base (Squeeze/Breath start)
        return base


    
    def update(_frame=0):
        global current_view, desired_view, current_state, event_detected, ignore_ui_callbacks
        global last_motion_log, motion_lock_expiry, motion_prev_ta, motion_prev_time
        global headset_on_head, LOG_WINDOW_HEIGHT
        global calib_mode, calib_phase, calib_step, calib_start_time, calib_step_start_time, calib_base_ta, calib_min_ta, calib_vals, last_calib_ratio
        global recording_start_time_obj, is_recording, session_start_ta, pending_rec
        global active_event_label, process_waiting_for_calib
        global first_run_center # [FIX] Add global
        global GSR_CENTER_VAL, GSR_CENTER_TARGET # [FIX] Add globals for smooth dampening
        global graph_bg, bg_scores, bg_count, bg_detail, bg_status, bg_sens, bg_info, bg_scale_panel, bg_system_panel
        

        if not first_run_center and latest_gsr_ta > 0.1:
             #print(f"[Auto-Center] First Reading: {latest_gsr_ta}")
             update_gsr_center(latest_gsr_ta)
             first_run_center = True
        
        current_state = "Audio/GSR Mode"
        
        ax_graph.grid(True, alpha=0.3)
        ax_graph.get_yaxis().set_visible(False)
        
        if is_recording: rec_text.set_alpha(1.0)
        else: rec_text.set_alpha(0.0)
        
        # [FIX] Init Loop Variables
        h_data = {}
        analyzer_res = {}
        msg = ""
        
        if current_view != desired_view:
            if desired_view == 'settings': 
                teleport_off(main_view_axes); teleport_on(settings_view_axes)
                # [NEW] Initialize mic name display with current selection
                if audio_handler and audio_handler.current_mic_name:
                    ui_refs['text_mic_name'].set_text(audio_handler.current_mic_name)
                else:
                    ui_refs['text_mic_name'].set_text("NO MIC")
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
                 except Exception: pass
                 finally: ignore_ui_callbacks = False
        
        # [REMOVED HRM LOGIC]
        # System Line Part Removed (system_line)
        
        # [NEW] Rolling TA Set (10s Median - Matches Visible Graph)
        if latest_gsr_ta > 0.1 and not calib_mode:
            history_rolling = list(bands_history['GSR'])[-600:]
            if len(history_rolling) > 60:
                rolling_median = float(np.median(history_rolling))
                
                # Determine "Target Center"
                log_curr = math.log10(max(0.01, latest_gsr_ta))
                eff_win = get_effective_window()
                log_center = math.log10(max(0.01, GSR_CENTER_VAL))
                min_p_log = log_center - (0.625 * eff_win)
                rel_y = (log_curr - min_p_log) / eff_win * 100.0
                
                log_smooth = math.log10(max(0.01, rolling_median))
                rel_y_smooth = (log_smooth - min_p_log) / eff_win * 100.0
                
                target_center = rolling_median
                reason = "System"
                
                if rel_y_smooth > 92 or rel_y_smooth < 8:
                    target_center = latest_gsr_ta
                    reason = "Edge_Pull"
                
                # [MOD] Set TARGET, actual dampening happens below
                update_gsr_center(target_center, reason=reason)
        
        # [NEW] Unified Dampening Loop (Runs Every Frame, even in Calib)
        # 5% step per frame towards wherever the target is.
        if abs(GSR_CENTER_TARGET - GSR_CENTER_VAL) > 0.0001:
             GSR_CENTER_VAL += (GSR_CENTER_TARGET - GSR_CENTER_VAL) * 0.05
            
        # [NEW] Update GSR UI
        # 1. Update Text
        txt_ta_score.set_text(f"INST TA: {latest_gsr_ta:.3f}")
        txt_count_val.set_text(f"TA Counter: {ta_accum:.2f}") 
        
        # [NEW] SYNC SENSITIVITY DISPLAY (Fixes "Never Changed" bug)
        # This ensures calibration updates and zoom updates are always shown.
        update_zoom_ui()
        
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
                     # [MOD] Removed redundant central motion text per user request
                     # if 'txt_motion_overlay' in ui_refs:
                     #     ui_refs['txt_motion_overlay'].set_text(overlay_text)
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
             # Scale/Sens/Span Text is now handled by update_zoom_ui() called earlier in update()
             pass

             # TA SET Label
             if 'txt_ta_set_line' in ui_refs:
                  ui_refs['txt_ta_set_line'].set_text(f"TA SET: {GSR_CENTER_VAL:.2f}")

             # [NEW] Update Grid Labels (Log Space)
             # 1 Unit = 20% of Window
             unit_log = 0.2 * eff_win # eff_win is log height here
             log_center = math.log10(max(0.01, GSR_CENTER_VAL))
             
             # Hardcoded keys based on initialization
             for u_idx in [2, 1, -1, -2, -3]:
                 key = f'txt_grid_{u_idx}'
                 if key in ui_refs:
                     target_log = log_center + (u_idx * unit_log)
                     target_ta = math.pow(10, target_log)
                     delta_ta = target_ta - GSR_CENTER_VAL
                     sign = "+" if delta_ta > 0 else ""
                     ui_refs[key].set_text(f"{sign}{delta_ta:.2f} TA")

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
                       if pat == "BLOWDOWN": col = '#00CED1' # DarkTurquoise
                       elif pat == "ROCKET READ": col = '#DC143C'  # Crimson
                       elif pat == "LONG FALL": col = '#008000'
                       elif pat == "SHORT FALL": col = '#3CB371'
                       elif pat == "LONG RISE": col = '#FF4500'    # OrangeRed
                       elif pat == "SHORT RISE": col = 'orange'
                       elif pat == "MOTION": col = '#8B0000'       # Dark Red (Motion Locked)
                       
                       disp_pat = f"{pat}" if pat else ""
                       ui_refs['txt_pattern'].set_text(disp_pat)
                       ui_refs['txt_pattern'].set_color(col)
                       
             
             # [NEW] Check for Auto-Start Recording (Triggered by 'R' key in background?)
             # Implement User Keybinds? 
             pass   # [OPT] Only update visible lines.
             
             for k, graph_line in lines.items():
                 if not graph_line.get_visible(): continue
                                 
                 # [REQ] Skip GSR if not connected
                 if k == 'GSR':
                      gsr_ok = False
                      if 'gsr_thread' in globals() and gsr_thread:
                           if gsr_thread.connected: gsr_ok = True
                      
                      if not gsr_ok:
                           if len(graph_line.get_xdata()) > 0: graph_line.set_data([], [])
                           continue
                 
                 data = bands_history[k]
                 if len(data) > 0:
                     # [OPT] Downsample (Decimation)
                     # [FIX] Thread-Safety: list(data) snapshots the deque to prevent mutation during iteration
                     # Using stride of 2
                     raw_full = list(data)
                     raw_ys = np.array(raw_full[::2])
                     
                     if k == 'GSR':
                           # [NEW] Vectorized Logarithmic Normalization
                           log_raw_ys = np.log10(np.maximum(0.01, raw_ys))
                           log_center = math.log10(max(0.01, GSR_CENTER_VAL))
                           log_win = eff_win # get_effective_window returns log height
                           
                           min_p_log = log_center - (0.625 * log_win)
                           # span_p is log_win
                           ys = (log_raw_ys - min_p_log) / log_win * 100.0
                     else:
                         ys = raw_ys

                     # [FIX] Scale X-axis by stride (2)
                     graph_line.set_data(np.arange(len(ys)) * 2, ys)
                 else:
                     graph_line.set_data([], [])

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

        # [REMOVED OLD AUTO-CENTER CHECK - Now handled by Rolling Median]

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
                        # [REQ] Restore Steadiness Calculation for input to helper
                        history = list(bands_history['GSR'])[-60:]
                        is_steady = False
                        if len(history) > 30:
                             rng = max(history) - min(history)
                             if rng < 0.15: is_steady = True

                        # [NEW] Refactored SQUEEZE Handler
                        phases = handle_calibration_phase(calib_step, calib_phase, calib_start_time, calib_base_ta, calib_min_ta, latest_gsr_ta, elapsed, is_steady, calib_step_start_time)
                        calib_phase, calib_min_ta, evt, msg, calib_base_ta, calib_start_time = phases
                        
                        # Handle specific Squeeze events if needed
                        if evt and "DROP" in evt: active_event_label = f"SQUEEZE_{calib_step}_DROP"
                        if evt and "RELEASE" in evt: active_event_label = f"SQUEEZE_{calib_step}_RELEASE"
                        
                        if calib_phase == 3:
                             # Wait for 1.5s stability
                             if time.time() - calib_start_time > 1.5:
                                  calib_phase = 2 # Go to Calculation
                                  
                                  total_drop = calib_base_ta - calib_min_ta
                                  if total_drop < 0.05: total_drop = 0.05
                                  calib_vals.append(total_drop)
                                  
                                  log_msg(f"Calib {calib_step}: Collected Drop={total_drop:.4f}")
                                  active_event_label = f"SQUEEZE_{calib_step}_COLLECTED"
                                 
                                  # [REQ] Log Median Logic at End of Step 3
                                  if calib_step == 3:
                                       # Compute Median
                                       median_drop = sorted(calib_vals)[1] # Middle of 3
                                       
                                       # NEW LOG FORMULA (Dynamic per User TA)
                                       # Target: The drop defined by median_drop must equal 45% of screen height
                                       # Use actual calib_base_ta (the level at the start of the squeeze)
                                       base_ta = max(0.1, calib_base_ta)
                                       log_base = math.log10(base_ta)
                                       log_drop_point = math.log10(max(0.05, base_ta - median_drop))
                                       log_drop_mag = abs(log_base - log_drop_point)
                                       
                                       # LOG_WINDOW_HEIGHT = Mag / 45%
                                       new_log_win = log_drop_mag / 0.45
                                       LOG_WINDOW_HEIGHT = max(0.01, min(2.5, new_log_win))
                                       
                                       log_msg(f"Calibration Complete: Median Drop={median_drop:.4f} -> LogWindow={LOG_WINDOW_HEIGHT:.4f}")
                                       active_event_label = "CALIB_COMPLETE_LOG"
                                       
                                       # Update Display
                                       update_gsr_center(latest_gsr_ta, reason="Calib (Final)")

                                  # [REQ] Reset TA (Center Graph)
                                  update_gsr_center(latest_gsr_ta, reason="Calib (Step Complete)")
                             
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
                            if time.time() - calib_start_time < 2.0:
                                 msg = "CALIB 4/4: RELAX HAND..."
                                 
                                 # [REQ] Hysteresis Tracking (Same as Squeeze)
                                 calib_base_ta = update_hysteresis(latest_gsr_ta, calib_base_ta, 0.05)
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
                                # Validate Drop (Log Ratio)
                                total_drop = calib_base_ta - calib_min_ta
                                # [MOD] Correct log-drop calculation
                                log_drop = abs(math.log10(max(0.1, calib_base_ta)) - math.log10(max(0.1, calib_min_ta)))
                                ratio = log_drop / LOG_WINDOW_HEIGHT
                                last_calib_ratio = ratio # [DEBUG]
                                
                                # [FIX] Strict Range Removed. Any valid drop is OK.
                                # Success
                                calib_phase = 2
                                calib_start_time = time.time()                                    
                                update_gsr_center(latest_gsr_ta, reason="Calib (Breath)")
                                active_event_label = f"BREATH_VERIFIED | Drop={total_drop:.4f}"
                                
                        elif calib_phase == 2:
                            msg = "CALIB: COMPLETE!"
                            if time.time() - calib_start_time > 2.0: # Show Success for 2s
   
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
                                    
                                    # [NEW] Process Hook
                                    if process_waiting_for_calib:
                                        # [FIX] Do NOT advance here if we are entering Phase 3 (Session Started)
                                        # The hook should only fire once. We'll do it in Phase 3 if pending_rec was True,
                                        if not pending_rec:
                                            process_waiting_for_calib = False
                                            advance_process_step()

                        elif calib_phase == 3:
                             msg = "SESSION STARTED"
                             session_start_ta = latest_gsr_ta # [FIX] Capture TA once session actually starts
                             if not counting_active: toggle_count(None)
                             
                             if time.time() - calib_start_time > 2.0:
                                  calib_mode = False
                                  update_gsr_center(latest_gsr_ta, force_pivot=True) # [REQ] Force Pivot Update
                                  # if gsr_patterns: gsr_patterns.reset() # [FIX] Disabled
                                  active_event_label = "SESSION_STARTED" # [REQ] Log Event
                                  if ovl: ovl.set_text("")
                                  log_msg("Calibration Sequence Finished")
                                  
                                  # [NEW] Process Hook
                                  if process_waiting_for_calib:
                                      process_waiting_for_calib = False
                                      advance_process_step()

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
                            
                except Exception as ex:
                    print(f"CALIB ERROR: {ex}")
                    calib_mode = False # Abort
                
             else:
                 # Hide
                 if 'txt_calib_overlay' in ui_refs:
                     ui_refs['txt_calib_overlay'].set_text("")
                     ui_refs['txt_calib_overlay'].set_visible(False)

             # [NEW] specific update for Session Detail Panel
             if is_recording and recording_start_time_obj:
                 elapsed = datetime.now() - recording_start_time_obj
                 total_sec = int(elapsed.total_seconds())
                 hours, remainder = divmod(total_sec, 3600)
                 mins, secs = divmod(remainder, 60)
                 
                 if 'txt_sess_len' in ui_refs:
                     ui_refs['txt_sess_len'].set_text(f"Duration : {hours:02}:{mins:02}:{secs:02}")
                 # Ensure color is black
                 if 'txt_sess_len' in ui_refs: ui_refs['txt_sess_len'].set_color('white')
              
             else:
                 pass 
             


        elif current_view == 'settings':
             pass



    def on_close(event):
        global app_running, sess_viewer, current_view
        if not app_running: return 
        app_running = False
        print("Window Close Event Triggered.")
        
        # [NEW] Clean up session viewer if active
        if current_view == 'viewer' and sess_viewer:
            try:
                sess_viewer.is_playing = False
                sess_viewer.stop_playback(reset=True)
                if sess_viewer.timer_id:
                    sess_viewer.master.after_cancel(sess_viewer.timer_id)
                    sess_viewer.timer_id = None
                # Close matplotlib figure
                if hasattr(sess_viewer, 'fig') and sess_viewer.fig:
                    plt.close(sess_viewer.fig)
            except: pass
        
        try:
             # Stop timer immediately
             if 'timer' in locals() and timer:
                 timer.stop()
        except: pass

    def on_resize(event):
        global graph_bg, bg_left_labels, bg_scores, bg_count, bg_detail, bg_status, bg_sens, bg_scale_panel, bg_system_panel
        graph_bg = bg_left_labels = bg_scores = bg_count = bg_detail = bg_status = bg_sens = bg_scale_panel = bg_system_panel = None
        # print("Resize: Backgrounds Cleared.")

    fig.canvas.mpl_connect('resize_event', on_resize)
    fig.canvas.mpl_connect('resize_event', on_resize)
    fig.canvas.mpl_connect('close_event', on_close)
    
    # [NEW] Keyboard Event Handler for Fullscreen Control
    def on_key(event):
        try:
            # 'Q' Trigger for Closing Questions
            if event.key.lower() == 'q':
                 if active_process_name or process_in_closing_phase:
                      trigger_closing_sequence()
                 return

            if event.key == 'escape':
                # If in a process and NOT yet in closing phase/end phase, ESC -> Closing
                if (active_process_name is not None) and (not process_in_closing_phase) and (not process_ending_phase):
                     print("[System] Escape pressed -> Triggering Closing Sequence")
                     trigger_closing_sequence()
                     return
                
                # Otherwise, Normal Exit
                print("[System] Escape key pressed. Exiting...")
                on_close(None)
                global app_running
                app_running = False  
        except Exception: pass

    fig.canvas.mpl_connect('key_press_event', on_key)

    # [FIX] Flush Last Log to UI
    if log_messages:
         try: log_msg(log_messages[-1].split("] ", 1)[-1])
         except Exception: pass

    try:
        # [FIX] Custom Animation Loop (Force 50fps)
        timer = fig.canvas.new_timer(interval=20) 
        
        # [NEW] User Selection (Wait until all functions/UI are defined)
        show_user_selection_dialog()

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
                 except Exception: pass
                 return

            if current_view != 'main': return
            
            # 1. Restore ALL backgrounds first
            if 'graph_bg' in globals() and graph_bg: fig.canvas.restore_region(graph_bg)
            if 'bg_left_labels' in globals() and bg_left_labels: fig.canvas.restore_region(bg_left_labels)
            if 'bg_scores' in globals() and bg_scores: fig.canvas.restore_region(bg_scores)
            if 'bg_count' in globals() and bg_count: fig.canvas.restore_region(bg_count)
            if 'bg_detail' in globals() and bg_detail: fig.canvas.restore_region(bg_detail)
            if 'bg_status' in globals() and bg_status: fig.canvas.restore_region(bg_status)
            if 'bg_sens' in globals() and bg_sens: fig.canvas.restore_region(bg_sens)
            if 'bg_scale_panel' in globals() and bg_scale_panel: fig.canvas.restore_region(bg_scale_panel)
            if 'bg_system_panel' in globals() and bg_system_panel: fig.canvas.restore_region(bg_system_panel)

            # 2. Draw ALL dynamic content
            if line: ax_graph.draw_artist(line)
            if 'txt_pattern' in ui_refs: ax_graph.draw_artist(ui_refs['txt_pattern'])
            
            if 'txt_ta_set_line' in ui_refs: ax_left_labels.draw_artist(ui_refs['txt_ta_set_line'])
            for u_idx in [2, 1, -1, -2, -3]:
                key = f'txt_grid_{u_idx}'
                if key in ui_refs: ax_left_labels.draw_artist(ui_refs[key])
                
            for s in ax_graph.spines.values(): ax_graph.draw_artist(s)
            
            try: ax_scores.draw_artist(txt_ta_score)
            except NameError: pass
            try: ax_scores.draw_artist(val_txt)
            except NameError: pass
            
            try: ax_count_bg.draw_artist(txt_count_val)
            except NameError: pass
            
            if 'txt_gsr_status' in ui_refs: ax_status.draw_artist(ui_refs['txt_gsr_status'])
            try: ax_status.draw_artist(txt_audio)
            except NameError: pass
            try: ax_status.draw_artist(rec_text)
            except NameError: pass
            
            if 'txt_sess_len' in ui_refs: ax_detail.draw_artist(ui_refs['txt_sess_len'])
            if 'txt_sess_date' in ui_refs: ax_detail.draw_artist(ui_refs['txt_sess_date'])
            if 'txt_sess_user' in ui_refs: ax_detail.draw_artist(ui_refs['txt_sess_user'])
            if 'txt_sess_time' in ui_refs: ax_detail.draw_artist(ui_refs['txt_sess_time'])

            if 'txt_calib_overlay' in ui_refs and ui_refs['txt_calib_overlay'].get_visible():
                ax_graph.draw_artist(ui_refs['txt_calib_overlay'])
            if 'txt_motion_overlay' in ui_refs and ui_refs['txt_motion_overlay'].get_visible():
                ax_graph.draw_artist(ui_refs['txt_motion_overlay'])
            if 'txt_process_overlay' in ui_refs and ui_refs['txt_process_overlay'].get_visible():
                ax_graph.draw_artist(ui_refs['txt_process_overlay'])

            # Draw Buttons
            for b_key in ['btn_count', 'btn_reset', 'btn_ta_set_now', 'btn_rec', 'btn_processes', 'btn_to_settings', 'btn_back', 'btn_manual', 'btn_viewer', 'btn_calib', 'btn_exit']:
                 if b_key in ui_refs and ui_refs[b_key].ax.get_visible():
                      b = ui_refs[b_key]
                      b.ax.draw_artist(b.ax.patch)
                      b.ax.draw_artist(b.label)
            
            # Draw Zoom Slider
            if ax_w_val: 
                ax_w_val.draw_artist(ax_w_val.patch)
                ax_w_val.draw_artist(txt_win_val)
                if 'txt_span_val' in ui_refs: ax_w_val.draw_artist(ui_refs['txt_span_val'])
            
            if 'ax_zoom_track' in ui_refs:
                 zt = ui_refs['ax_zoom_track']
                 zt.draw_artist(zt.patch)
                 zt.draw_artist(ui_refs['rect_track'])
                 for t in zt.texts: zt.draw_artist(t)
                 for l in zt.lines: zt.draw_artist(l)
                 zt.draw_artist(ui_refs['thumb_zoom'])

            # 3. Blit ALL modified regions in one pass
            if ax_graph: fig.canvas.blit(ax_graph.bbox)
            if ax_left_labels: fig.canvas.blit(ax_left_labels.bbox)
            if ax_scores: fig.canvas.blit(ax_scores.bbox)
            if ax_count_bg: fig.canvas.blit(ax_count_bg.bbox)
            if ax_w_val: fig.canvas.blit(ax_w_val.bbox)
            if ax_status: fig.canvas.blit(ax_status.bbox)
            if ax_detail: fig.canvas.blit(ax_detail.bbox)
            
            if ax_ctrl_bg: fig.canvas.blit(ax_ctrl_bg.bbox)
            if 'ax_system_bg' in ui_refs: fig.canvas.blit(ui_refs['ax_system_bg'].bbox)
            
            if 'ax_zoom_track' in ui_refs: fig.canvas.blit(ui_refs['ax_zoom_track'].bbox)
            if 'ax_calib' in ui_refs: fig.canvas.blit(ui_refs['ax_calib'].bbox)
            if 'btn_rec' in ui_refs: fig.canvas.blit(ui_refs['btn_rec'].ax.bbox)
            if 'btn_processes' in ui_refs and ui_refs['btn_processes'].ax.get_visible(): fig.canvas.blit(ui_refs['btn_processes'].ax.bbox)
            if 'btn_to_settings' in ui_refs and ui_refs['btn_to_settings'].ax.get_visible(): fig.canvas.blit(ui_refs['btn_to_settings'].ax.bbox)
            if 'btn_manual' in ui_refs and ui_refs['btn_manual'].ax.get_visible(): fig.canvas.blit(ui_refs['btn_manual'].ax.bbox)
            if 'btn_viewer' in ui_refs and ui_refs['btn_viewer'].ax.get_visible(): fig.canvas.blit(ui_refs['btn_viewer'].ax.bbox)
            if 'btn_back' in ui_refs and ui_refs['btn_back'].ax.get_visible(): fig.canvas.blit(ui_refs['btn_back'].ax.bbox)
            if 'btn_exit' in ui_refs and ui_refs['btn_exit'].ax.get_visible(): fig.canvas.blit(ui_refs['btn_exit'].ax.bbox)
            
            fig.canvas.flush_events()

            
            
            
            

            
            
        # Monkey patch update to call final_blit
        original_update = update
        def update_wrapper(frame=0):
             if not app_running: return
             
             # [NEW] One-shot Fullscreen/Maximize
             global first_run_zoomed
             global first_run_zoomed
             if 'first_run_zoomed' not in globals():
                 first_run_zoomed = True
                 try:
                     mngr = plt.get_current_fig_manager()
                     try: 
                         # True Fullscreen Mode
                         mngr.window.attributes('-fullscreen', True)
                     except: 
                         mngr.window.state('zoomed') # Windows fallback
                 except: pass
             try:
                 original_update(frame)
                 final_blit()
             except Exception:
                 # Silently ignore Matplotlib/Tkinter errors during shutdown
                 if not app_running: pass
                 else: raise
             
        # Re-bind timer
        timer.remove_callback(update)
        timer.add_callback(update_wrapper)
        plt.show()
    except KeyboardInterrupt:
        print("\n[System] Stopped by User (KeyboardInterrupt).")
    except Exception as e:
        import traceback
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        app_running = False
        print("Shutdown Initiated.")
        
        # 1. Immediate Timer Stop
        try:
             if 'timer' in locals() and timer:
                 timer.stop()
        except Exception: pass
        
        # 2. Cleanup Viewer if active
        if 'sess_viewer' in globals() and sess_viewer:
             try: sess_viewer.request_close()
             except: pass

        # 3. Stop GSR Thread
        if 'gsr_thread' in globals() and gsr_thread:
            print("[System] Stopping GSR Reader...")
            gsr_thread.stop()
            gsr_thread.join(timeout=0.5)

        # 4. Stop Recording & Audio
        if is_recording:
             try: toggle_rec(None) 
             except: pass
             
        if 'audio_handler' in globals() and audio_handler:
             try:
                 audio_handler.stop_recording()
                 audio_handler.stop_playback()
                 if audio_handler.audio_stream:
                      audio_handler.audio_stream.close()
             except: pass

        # 5. Final Cleanup
        save_config() 
        if f_gsr: 
             try: f_gsr.close()
             except Exception: pass

        print("Shutdown Complete.")
        os._exit(0)
