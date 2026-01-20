
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
CONFIG_FILE = "v30_config.json"

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
gsr_capture_queue = collections.deque(maxlen=100)

def get_effective_window():
    # [REQ] Use active_boost_level (applied on Reset) not target booster_level
    if active_boost_level == 0: return BASE_SENSITIVITY
    
    # Logic from v20
    mult = [1.0, 0.6, 1.0, 1.4][active_boost_level]
    safe_ta = max(1.0, GSR_CENTER_VAL)
    try:
        # [FIX] Invert ratio: High TA = Larger Variance = Needs Larger Window
        # Old: (2.0 / safe_ta) -> Shrinks window at High TA -> Instability
        # New: (safe_ta / 2.0) -> Expands window at High TA -> Stability
        return BASE_SENSITIVITY * math.pow((safe_ta / 2.0), mult)
    except:
        return BASE_SENSITIVITY


# --- UUIDS ---
UUID_BATTERY = "00002a19-0000-1000-8000-00805f9b34fb"
UUID_MODEL_NUMBER = "00002a24-0000-1000-8000-00805f9b34fb"
UUID_SERIAL = "00002a25-0000-1000-8000-00805f9b34fb"
UUID_FIRMWARE = "00002a26-0000-1000-8000-00805f9b34fb"
UUID_HARDWARE = "00002a27-0000-1000-8000-00805f9b34fb"
UUID_MANUFACTURER = "00002a29-0000-1000-8000-00805f9b34fb"
UUID_WRITE = "0d740002-d26f-4dbb-95e8-a4f5c55c57a9"
UUID_NOTIFY = "0d740003-d26f-4dbb-95e8-a4f5c55c57a9"



# --- COMMAND CONSTANTS ---
CMD_2_CONFIG = bytearray.fromhex("434d534e000c08021208080910ffc9b9f306504b4544")
CMD_3_STREAM = bytearray.fromhex("434d534e0007080312030a0103504b4544")
CMD_4_SETUP = bytearray.fromhex("434d534e0007080412030a010e504b4544")
CMD_5_FINAL = bytearray.fromhex("434d534e000608051202080d504b4544")

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

# Filters
b_filt, a_filt = butter(4, [1.0 / (FS / 2), 100.0 / (FS / 2)], btype='band')
zi_ch1 = lfilter_zi(b_filt, a_filt)
tx_seq = 1

# --- AUDIO RECORDING STATE ---
AUDIO_FS = 44100
audio_buffer = [] 
audio_stream = None
audio_filename = None
selected_device_idx = None 
current_mic_gain = 3.0 
current_mic_name = "Default"
audio_state = {'peak': 0.0, 'debug_ctr': 0} 

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
                                  writer_gsr.writerow([ts_now, f"{self.current_ta:.5f}", f"{GSR_CENTER_VAL:.3f}", f"{1.0/get_effective_window():.3f}", 0])
                             
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
                                  
                                  if has_data: 
                                       writer_trend.writerow(t_row)
                                       
                         except Exception as e:
                             pass

                    # [NEW] GRAPH HISTORY UPDATE (Master Clock = 60Hz)
                    try:
                         # 1. GSR Value
                         # [FIX] Store RAW TA for dynamic scaling in main loop
                         bands_history['GSR'].append(self.current_ta)
                         
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

def notification_handler(sender, data):
    global headset_on_head, current_state, is_recording, current_calm_score
    global samples_since_contact, last_packet_time, zi_ch1, eeg_buffer
    global last_on_signal_time, total_samples_recorded
    global current_gsr_display_val
    
    last_packet_time = time.time()
    if not data.startswith(b'CMSN'): return
    if timestamp_queue is not None:
        try: timestamp_queue.put_nowait(data)
        except: pass

    body = data[6:]
    def parse_payload(payload):
        global device_battery_level
        idx = 0
        extracted = []
        while idx < len(payload):
            key = payload[idx]; idx += 1
            if key in [0x12, 0x32]:
                if idx >= len(payload): break
                length = payload[idx]; idx += 1
                if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                if idx + length <= len(payload):
                    extracted.extend(parse_payload(payload[idx:idx + length]))
                idx += length
            elif key == 0x22: 
                if idx >= len(payload): break
                length = payload[idx]; idx += 1
                if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                if idx + length <= len(payload):
                    raw_bytes = payload[idx:idx + length]
                    for i in range(0, len(raw_bytes), 3):
                        chunk = raw_bytes[i:i + 3]
                        if len(chunk) == 3: extracted.append(int.from_bytes(chunk, byteorder='big', signed=True))
                idx += length
            elif key == 0x18: 
                if idx < len(payload):
                    val = payload[idx]; update_status_globals(val); idx += 1
            elif key == 0x08: 
                # [FIX] Sequence Number is VarInt (Sync with v21)
                while idx < len(payload):
                     if idx >= len(payload): break
                     byte = payload[idx]; idx += 1
                     if not (byte & 0x80): break
            elif key == 0x10: idx += 1
            elif key == 0x01: 
                # [FIX] Battery Level Parsing
                if idx < len(payload):
                    val = payload[idx]
                    device_battery_level = f"{val}%"
                    idx += 1
            elif key > 0x20:
                if idx >= len(payload): break
                length = payload[idx]; idx += 1
                if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                idx += length
            else: idx += 1
        return extracted

    def update_status_globals(val):
        global headset_on_head, last_on_signal_time
        now = time.time()
        if val == 1:
            last_on_signal_time = now
            if not headset_on_head: headset_on_head = True; log_msg("Sensor: ON HEAD")
        elif val == 2:
            if now - last_on_signal_time > 1.5:
                if headset_on_head: headset_on_head = False; log_msg("Sensor: OFF HEAD")

    try:
        new_raw = []
        try:
           new_raw = parse_payload(body)
        except Exception: 
           pass
           
        if len(new_raw) > 0:
            filt_chunk, zi_ch1 = lfilter(b_filt, a_filt, new_raw, zi=zi_ch1)
            current_state = "EEG: STREAMING"
            detailed_rows = []
            for i, val in enumerate(filt_chunk):
                # [Refactor] Push to Queue for High-Res Processing
                try: raw_eeg_queue.put_nowait(val)
                except: pass
                
                # eeg_buffer.append(val) <--- Moved to Consumer
                if headset_on_head: samples_since_contact += 1
                else: samples_since_contact = 0
                is_warmed_up = samples_since_contact > WARMUP_SAMPLES
                req_samples = int(FS * FFT_WINDOW_SEC)
                bands = [0]*5; calm_inst=0
                
                # Initialize smoothed_bands to avoid ReferenceError if loop not entered
                smoothed_bands = [0]*5
                
                # [Refactor] FFT Moved to GSR Thread (60Hz Polling)
                # notification_handler only fills eeg_buffer now.
                     


                


    except Exception as e: log_msg(f"Process Err: {e}")

def save_audio():
    global audio_buffer, audio_filename
    if not audio_buffer or not audio_filename:
        return
    try:
        if len(audio_buffer) > 0:
            full_recording = np.concatenate(audio_buffer, axis=0)
            wav.write(audio_filename, AUDIO_FS, full_recording)
            log_msg(f"Audio Saved: {audio_filename}")
        else:
             log_msg("Audio Buffer Empty")
    except Exception as e:
        log_msg(f"Audio Save Error: {e}")
    finally:
        audio_buffer = []  

def fragment_packet(data, chunk_size=20):
    chunks = []
    for i in range(0, len(data), chunk_size): chunks.append(data[i:i + chunk_size])
    return chunks

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
            'mic_name': current_mic_name,
            'mic_rate': current_mic_rate if 'current_mic_rate' in globals() else None, # [NEW]
            'mic_gain': current_mic_gain,
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
    global current_mic_name, current_mic_gain
    global DEVICE_ADDRESS, ADVERTISED_NAME
    
    if not os.path.exists(CONFIG_FILE):
        print("[Config] No file found. Using defaults.")
        return

    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            
        current_mic_name = cfg.get('mic_name', "Default")
        current_mic_gain = float(cfg.get('mic_gain', 3.0))
        
        # [new] Restore Mic Rate
        global current_mic_rate
        current_mic_rate = cfg.get('mic_rate', None)
        
        # Restore Theme
        global current_theme
        current_theme = cfg.get('gui_theme', 'Light')
        
        # Restore GSR Settings
        global GSR_CENTER_VAL, BASE_SENSITIVITY, booster_level
        GSR_CENTER_VAL = float(cfg.get('gsr_center', 3.0))
        BASE_SENSITIVITY = float(cfg.get('gsr_base', 0.3)) # [FIX] Fallback to 'gsr_window' if missing? 
        if 'gsr_base' not in cfg: BASE_SENSITIVITY = float(cfg.get('gsr_window', 0.3))
            
        booster_level = int(cfg.get('booster_idx', 0))
        
        # Restore Graph Visibility
        global initial_graph_visibility
        initial_graph_visibility = cfg.get('graph_visibility', None)
        
        # Restore Bluetooth
        saved_addr = cfg.get('bt_address', None)
        saved_name = cfg.get('bt_name', "Unknown")
        if saved_addr:
             DEVICE_ADDRESS = saved_addr
             ADVERTISED_NAME = saved_name
             print(f"[Config] Restored Target Device: {ADVERTISED_NAME} ({DEVICE_ADDRESS})")
        
        print(f"[Config] Loaded settings. Mic: {current_mic_name}, Rate: {current_mic_rate}, Gain: {current_mic_gain}")
    except Exception as e:
        print(f"[Config] Load Error: {e}")

def encode_varint(v):
    p = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v: p.append(b | 0x80)
        else: p.append(b); break
    return p

def create_auth_packet(auth_seq=1, auth_type=1):
    serial_bytes = REAL_SERIAL_STR.encode('utf-8')
    inner = bytearray([0x08, auth_type, 0x32, len(serial_bytes)]) + serial_bytes
    seq_bytes = encode_varint(auth_seq)
    outer = bytearray([0x08]) + seq_bytes + bytearray([0x12, len(inner)]) + inner
    pkt = bytearray.fromhex("434d534e") + bytearray([0x00, len(outer)]) + outer + bytearray.fromhex("504b4544")
    return pkt

def create_timestamp_packet():
    ts_ms = int(time.time() * 1000)
    ts_varint = encode_varint(ts_ms)
    inner = bytearray([0x08, 0x09, 0x10]) + ts_varint
    outer = bytearray([0x08, 0x02, 0x12, len(inner)]) + inner
    pkt = bytearray.fromhex("434d534e") + bytearray([0x00, len(outer)]) + outer + bytearray.fromhex("504b4544")
    return pkt

def send_command(key, payload=None):
    global tx_seq
    if not is_connected: return
    try:
        inner = bytearray([0x08, key]) 
        if payload: inner += payload
        seq_bytes = encode_varint(tx_seq)
        outer = bytearray([0x08]) + seq_bytes + bytearray([0x12, len(inner)]) + inner
        tx_seq += 1
        pkt = bytearray.fromhex("434d534e") + bytearray([0x00, len(outer)]) + outer + bytearray.fromhex("504b4544")
        command_queue.put(pkt)
    except Exception as e: log_msg(f"CMD Fail: {e}")

async def listener(client, rx_queue):
    timestamp_answered = False
    log_msg("Listener Started")
    while True:
        try:
            if not client.is_connected: break
            try: data = await asyncio.wait_for(rx_queue.get(), timeout=0.1)
            except asyncio.TimeoutError: continue
            if not timestamp_answered and b'\x18\x00' in data and b'\x32\x04\x10' in data:
                log_msg("TIMESTAMP REQUEST")
                ts_pkt = create_timestamp_packet()
                chunks = fragment_packet(ts_pkt)
                for c in chunks: await client.write_gatt_char(UUID_WRITE, c, response=False); await asyncio.sleep(0.05)
                timestamp_answered = True
        except: break

async def bluetooth_task():
    global is_connected, current_state, last_packet_time, device_battery_level, current_client, timestamp_queue
    global REAL_SERIAL_STR, DEVICE_ADDRESS, ADVERTISED_NAME # [FIX] Ensure REAL_SERIAL_STR is global
    
    STRATEGIES = [{"seq": 2, "type": 2, "delay": 1.0}, {"seq": 50, "type": 2, "delay": 1.0}, {"seq": 1, "type": 1, "delay": 2.0}]
    
    conn_attempts = 0 # [REQ] Limit attempts
    
    while True:
        try:
            # [REQ] Stop after 20 failed attempts
            if conn_attempts >= 5:
                 log_msg("Max Connection Attempts (5) Reached. Stopping Scan.")
                 break
            
            conn_attempts += 1
            log_msg(f"[EEG] Scanning for {ADVERTISED_NAME if ADVERTISED_NAME else 'Devices'}... (Attempt {conn_attempts}/5)")
            devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
            candidate = None
            
            if ADVERTISED_NAME and ADVERTISED_NAME != "Unknown":
                for d, adv in devices.values():
                    if d.name == ADVERTISED_NAME:
                        candidate = d; break
            
            if not candidate:
                for d, adv in devices.values():
                    if (d.name and "Brain" in d.name) or (d.name and "FC11" in d.name):
                        candidate = d; ADVERTISED_NAME = d.name; break
                    if "0d740001-d26f-4dbb-95e8-a4f5c55c57a9" in adv.service_uuids:
                        candidate = d; ADVERTISED_NAME = adv.local_name if adv.local_name else d.name; break
            
            if candidate: 
                DEVICE_ADDRESS = candidate.address
                log_msg(f"Found: {ADVERTISED_NAME} ({DEVICE_ADDRESS})")
            else: 
                log_msg("[EEG] Device Not Found. Retrying...")
                await asyncio.sleep(3.0); continue

            log_msg(f"Connecting {DEVICE_ADDRESS}...")
            async def safe_write(cl, data, delay=0.1):
                chunks = fragment_packet(data)
                for c in chunks: await cl.write_gatt_char(UUID_WRITE, c, response=False); await asyncio.sleep(0.05)
                await asyncio.sleep(delay)

            try:
                client = BleakClient(DEVICE_ADDRESS, timeout=10.0)
                await client.connect()
                if not client.is_connected: raise Exception("Conn Fail")
                log_msg("Connected.")
                conn_attempts = 0 # [REQ] Reset counter on success
                await asyncio.sleep(0.5) # [FIX] Allow BlueZ to settle (Critical for Service Discovery)
                
                # [FIX] v21 does not use get_services() here, relying on automatic discovery or cached services.
                # Removing the explicit call that crashed on older Bleak versions.
                
                # Read Device Info (Safe Mode)
                dev_info = {'name': ADVERTISED_NAME}
                
                async def read_safe(uuid, label):
                    try:
                        data = await client.read_gatt_char(uuid)
                        val = data.decode('utf-8').strip()
                        log_msg(f"Read {label}: {val}")
                        return val
                    except Exception as e:
                        # log_msg(f"Read {label} Fail: {e}")
                        return "?"

                dev_info['serial'] = await read_safe(UUID_SERIAL, "Serial")
                dev_info['manuf'] = await read_safe(UUID_MANUFACTURER, "Manuf")
                dev_info['fw'] = await read_safe(UUID_FIRMWARE, "FW")
                dev_info['hw'] = await read_safe(UUID_HARDWARE, "HW")
                
                ui_update_queue.put_nowait(dev_info)
                
                REAL_SERIAL_STR = dev_info['serial']
                
                current_client = client; timestamp_queue = asyncio.Queue()
                listen_task = asyncio.create_task(listener(client, timestamp_queue))

                handshake_done = False
                handshake_done = False
                for idx, strat in enumerate(STRATEGIES):
                    try:
                        log_msg(f"Strat {idx+1} Start: Seq={strat['seq']}")
                        
                        # 1. Auth Packet
                        auth_pkt = create_auth_packet(strat['seq'], strat['type'])
                        await safe_write(client, auth_pkt, delay=strat['delay'])
                        if not client.is_connected:
                             log_msg(f"Strat {idx+1} Disconnected after Auth.")
                             listen_task.cancel(); await client.disconnect(); await asyncio.sleep(1.0); await client.connect()
                             current_client = client; timestamp_queue = asyncio.Queue(); listen_task = asyncio.create_task(listener(client, timestamp_queue)); continue
                        
                        # 2. Config
                        log_msg(f"Strat {idx+1} Sending Config...")
                        await safe_write(client, CMD_2_CONFIG, delay=0.2)
                        
                        # 3. Notify
                        log_msg(f"Strat {idx+1} Starting Notify...")
                        await client.start_notify(UUID_NOTIFY, notification_handler)
                        await asyncio.sleep(0.5) # [FIX] Allow notify subscription to settle
                        
                        # 4. Stream & Setup
                        log_msg(f"Strat {idx+1} Setup & Stream...")
                        await safe_write(client, CMD_3_STREAM, delay=0.2)
                        await safe_write(client, CMD_4_SETUP, delay=0.2)
                        await safe_write(client, CMD_5_FINAL, delay=0.2)
                        
                        log_msg(f"Strat {idx+1} Success!")
                        handshake_done = True; break
                    except Exception as e: log_msg(f"Strat {idx+1} Fail: {e}")
                
                if not handshake_done: raise Exception("Handshake Failed")

                log_msg("Streaming...")
                is_connected = True; current_state = "EEG: STREAMING"
                
                try:
                    val = await client.read_gatt_char(UUID_BATTERY)
                    if len(val)>0: device_battery_level = f"{int(val[0])}%"; log_msg(f"Battery: {device_battery_level}")
                except: pass

                batt_timer = 0
                while client.is_connected and app_running:
                    while not command_queue.empty():
                         try:
                             pkt = command_queue.get_nowait()
                             await safe_write(client, pkt, delay=0.05) 
                             await asyncio.sleep(0.05) 
                         except: pass

                    await asyncio.sleep(0.1) 
                    batt_timer += 1
                    if batt_timer >= 100: 
                        try:
                            val = await client.read_gatt_char(UUID_BATTERY)
                            if len(val)>0: device_battery_level = f"{int(val[0])}%"
                        except: pass
                        batt_timer = 0
                
                listen_task.cancel()
            except Exception as e: 
                log_msg(f"Error: {e}")
                if "Conn Fail" in str(e) or "Serial" in str(e):
                     DEVICE_ADDRESS = None 
            finally:
                if 'client' in locals() and client.is_connected: await client.disconnect()
                if 'listen_task' in locals(): listen_task.cancel()
                
                # [FIX] Reset Status on Disconnect
                headset_on_head = False
                device_battery_level = "--"
                current_state = "EEG: DISCONNECTED"
                log_msg("[BLE] Disconnected. Status Reset.")
            if not is_connected: await asyncio.sleep(2.0)
        except: await asyncio.sleep(2.0)

def run_ble():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bluetooth_task())

if __name__ == "__main__":
    # Start GSR Thread
    gsr_thread = GSRReader()
    gsr_thread.start()
    
    t = threading.Thread(target=run_ble, daemon=True)

    t.start()
    
    load_config() 
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

    # [NEW] Set Point Dashed Line
    # Fixed at 62.5% because our dynamic scale always places GSR_CENTER_VAL at that visual position.
    # [REQ] Deep Orange and Label
    ax_graph.axhline(y=62.5, color='#CC5500', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    txt_ta_set_line = ax_graph.text(0, 63.0, f"TA SET: {GSR_CENTER_VAL:.2f}", color='#CC5500', fontsize=8, fontweight='bold', ha='left')
    ui_refs['txt_ta_set_line'] = txt_ta_set_line
    
    # [NEW] Calibration Overlay Text (Figure Level for Visibility)
    txt_calib_overlay = fig.text(0.5, 0.5, "", ha='center', va='center', fontsize=24, fontweight='bold', color='red', zorder=100)
    ui_refs['txt_calib_overlay'] = txt_calib_overlay # Store ref
    
    # === CHECKBUTTON LEGEND (Right Side) ===
    # Shifted Right: [0.85, ...]
    ax_check = reg_ax([0.85, 0.60, 0.10, 0.25], main_view_axes)
    ax_check.set_title("Display", fontsize=10, fontweight='bold')
    ax_check.set_facecolor('white')
    
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
    r_ctrl = [0.835, 0.26, 0.13, 0.33] # H = 0.33
    ax_ctrl_bg = reg_ax(r_ctrl, main_view_axes)
    ax_ctrl_bg.set_facecolor('#f9f9f9')
    ax_ctrl_bg.set_xticks([]); ax_ctrl_bg.set_yticks([])
    # Border
    rect_ctrl_border = plt.Rectangle((0,0), 1, 1, transform=ax_ctrl_bg.transAxes, fill=False, ec='#aaaaaa', lw=2, clip_on=False)
    ax_ctrl_bg.add_patch(rect_ctrl_border)
    
    # --- 1. Title: GSR Scale ---
    # Relative to main axes to align easily
    ax_scale_lbl = reg_ax([0.835, 0.54, 0.13, 0.04], main_view_axes)
    ax_scale_lbl.set_axis_off()
    ax_scale_lbl.text(0.5, 0.5, "GSR Scale", ha='center', fontweight='bold', fontsize=12)
    
    # --- 2. Sensitivity ---
    # Label
    ax_win_lbl = reg_ax([0.835, 0.50, 0.13, 0.03], main_view_axes)
    ax_win_lbl.set_axis_off()
    ax_win_lbl.text(0.5, 0.5, "Sensitivity", ha='center', va='center', fontsize=10, fontweight='bold', color='#444')
    
    # Stepper [ - ] [ Val ] [ + ]
    # Centered in 0.835 + 0.13 = range 0.835 to 0.965. Center ~ 0.90
    y_sens = 0.46
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
    ax_boost_lbl = reg_ax([0.835, 0.41, 0.13, 0.02], main_view_axes)
    ax_boost_lbl.set_axis_off()
    ax_boost_lbl.text(0.5, 0.5, "Auto-Boost", ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    # Buttons [OFF] [L] [M] [H]
    # Spread evenly: 4 buttons in 0.13 width.
    # ~0.03 per button gap?
    y_b = 0.37
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
        # active_boost_level = lvl # [REQ] Delay sync until Reset occurs
        
        update_boost_ui()
        # Update text immediately
        txt_win_val.set_text(f"{get_display_sens():.2f}")
        
        # [REQ] Reset TA and Graph when Auto-Boost changes
        if latest_gsr_ta > 0.01:
             update_gsr_center(float(f"{latest_gsr_ta:.2f}"))
             
        plt.draw()

    btn_b_off.on_clicked(lambda e: set_boost(0))
    btn_b_lo.on_clicked(lambda e: set_boost(1))
    btn_b_med.on_clicked(lambda e: set_boost(2))
    btn_b_hi.on_clicked(lambda e: set_boost(3))
    
    # --- 4. Calibrate ---
    # Centered at bottom of panel
    ax_calib = reg_ax([0.85, 0.28, 0.10, 0.04], main_view_axes)
    btn_calib = Button(ax_calib, "Calibrate", color='lightblue', hovercolor='cyan')
    
    # Saved Boost Level for Restore
    global saved_boost_level
    saved_boost_level = 0
    
    def start_calibration(e):
        global calib_mode, calib_step, calib_phase, calib_start_time, calib_base_ta, calib_min_ta, calib_vals
        global booster_level, saved_boost_level
        
        # Save current level
        saved_boost_level = booster_level
        
        # Force Manual Mode (OFF)
        set_boost(0) 
        
        calib_mode = True
        calib_phase = 0
        calib_step = 1
        calib_start_time = time.time()
        calib_base_ta = latest_gsr_ta
        calib_min_ta = latest_gsr_ta
        calib_vals = [] # Store drops
        
        log_msg(f"Calibration Started. Saving Boost Lvl: {saved_boost_level}")
        
    btn_calib.on_clicked(start_calibration)
    
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
        BASE_SENSITIVITY = max(0.05, min(10.0, rounded_step(new_win)))
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
        
        # [NEW] TA Counter Logic (Count Drops)
        if counting_active:
             diff = GSR_CENTER_VAL - val
             if diff > 0 and not calib_mode: # [REQ] Only count drops if not calibrating
                 ta_accum += diff
                 
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
    txt_conn = ax_status.text(0.12, 0.5, "EEG: ●", color='lightgray', fontsize=11, fontweight='bold', va='center')
    txt_wear = ax_status.text(0.25, 0.5, "HEADSET: --", color='lightgray', fontsize=11, fontweight='bold', va='center')
    txt_batt = ax_status.text(0.48, 0.5, "BATT: --", color='lightgray', fontsize=11, fontweight='bold', va='center')
    txt_audio = ax_status.text(0.65, 0.5, "AUDIO: --", color='lightgray', fontsize=11, fontweight='bold', va='center')
    
    # Rec moved to far right
    rec_text = ax_status.text(0.92, 0.5, "● REC", color='red', fontsize=11, fontweight='bold', va='center', visible=False)
    
    ui_refs['txt_gsr_status'] = txt_gsr_status

    # Record Button (Moved Left)
    r_rc = [0.05, 0.06, 0.12, 0.05] 
    ax_rec = reg_ax(r_rc, main_view_axes)
    ui_refs['btn_rec'] = Button(ax_rec, 'Record', color='lightgreen')
    
    import tkinter as tk

    # --- AUDIO STREAM LOGIC (SMART STREAM) ---
    def global_audio_callback(indata, frames, time, status):
        global audio_buffer, audio_state, current_mic_gain, is_recording
        gained_sig = indata * current_mic_gain
        peak = np.max(np.abs(gained_sig))
        audio_state['peak'] = peak 
        if is_recording:
             amplified = np.clip(gained_sig, -1.0, 1.0)
             audio_buffer.append(amplified.copy())

    def sync_audio_stream(target_view):
        global audio_stream, selected_device_idx, current_mic_gain, is_recording, audio_state, current_mic_name
        global current_mic_rate # [FIX] Added here to avoid SyntaxError
        
        should_be_on = is_recording or (target_view == 'settings')

        if selected_device_idx is None:
            try:
                devs = sd.query_devices()
                target_mic = current_mic_name
                selected_device_idx = None
                current_mic_name = "NO MIC" 
                
                if target_mic and target_mic != "Default" and target_mic != "N/A" and target_mic != "NO MIC":
                     for i, d in enumerate(devs):
                         if d['max_input_channels'] > 0 and (target_mic in d['name'] or d['name'] in target_mic):
                              try:
                                  with sd.InputStream(device=i, samplerate=None, channels=1) as s:
                                      s.read(10) # Must actually read data
                                  selected_device_idx = i
                                  current_mic_name = d['name'] 
                                  log_msg(f"Mic Verified (Target): {current_mic_name}")
                                  break
                              except Exception as pe:
                                  log_msg(f"Probe Fail (Target): {d['name']} ({pe})")
                
                if selected_device_idx is None:
                    for i, d in enumerate(devs):
                        if d['max_input_channels'] > 0:
                             try:
                                 with sd.InputStream(device=i, samplerate=None, channels=1) as s:
                                     s.read(10)
                                 selected_device_idx = i
                                 current_mic_name = d['name']
                                 log_msg(f"Mic Verified (Fallback): {current_mic_name}")
                                 break
                             except: pass

                if 'text_mic_name' in ui_refs:
                     ui_refs['text_mic_name'].set_text(current_mic_name)
                if 'txt_audio' in locals() or 'txt_audio' in globals(): # Safety
                     txt_audio.set_text(f"AUDIO: {current_mic_name[:15]}")
                     
            except Exception as e: log_msg(f"Dev Check Err: {e}")

        if not should_be_on:
            if audio_stream:
                audio_stream.stop(); audio_stream.close(); audio_stream = None
                log_msg("Audio Stream: OFF")
            return

        if audio_stream and audio_stream.active:
             return 
             
        try:
            if selected_device_idx is None: return 
            
            # [REQ] Prioritize saved sample rate if available
            saved_rate = None
            if 'current_mic_rate' in globals() and current_mic_rate:
                 saved_rate = current_mic_rate
                 
            # Construct candidate list: Saved (if any), 44100, 48000, 48k Alt, None (Auto)
            rates_to_try = []
            if saved_rate: rates_to_try.append(saved_rate)
            rates_to_try.extend([44100, 48000, None])
            
            # Deduplicate while preserving order
            rates_to_try = list(dict.fromkeys(rates_to_try))
            
            stream_created = False
            
            if audio_stream:
                 try: audio_stream.close()
                 except: pass
                 audio_stream = None
            time.sleep(0.2) 
            
            for sr in rates_to_try:
                try:
                    log_msg(f"Trying SR: {sr if sr else 'Auto'}...")
                    audio_stream = sd.InputStream(
                        samplerate=sr, channels=1, device=selected_device_idx, 
                        callback=global_audio_callback, blocksize=1024
                    )
                    audio_stream.start()
                    stream_created = True
                    actual_rate = sr if sr else audio_stream.samplerate
                    log_msg(f"Audio Stream: ON ({actual_rate} Hz)")
                    
                    # [REQ] Save working rate
                    current_mic_rate = int(actual_rate)
                    
                    break
                except Exception as e:
                    log_msg(f"SR {sr} Fail: {e}")
                    if audio_stream: 
                        try: audio_stream.close()
                        except: pass; 
                        audio_stream = None
                    time.sleep(0.3) 
            
            if not stream_created:
                raise Exception("All Sample Rates Failed")

        except Exception as e: log_msg(f"Stream Err: {e}")

    def open_audio_select(e):
         global selected_device_idx, current_mic_name, audio_stream
         
         root = tk.Tk()
         root.withdraw()
         devices = sd.query_devices()
         input_devices = []
         for i, d in enumerate(devices):
             if d['max_input_channels'] > 0:
                 input_devices.append(f"{i}: {d['name']}")
         
         if not input_devices:
             log_msg("No Input Devices Found!")
             root.destroy(); return
         
         dlg = tk.Toplevel(root)
         dlg.title("Select Microphone")
         dlg.geometry("400x150")
         tk.Label(dlg, text="Choose Input Device:").pack(pady=5)
         combo = ttk.Combobox(dlg, values=input_devices, width=50)
         
         idx_to_sel = 0
         if selected_device_idx is not None:
             for j, d_str in enumerate(input_devices):
                 if d_str.startswith(f"{selected_device_idx}:"):
                     idx_to_sel = j; break
         combo.current(idx_to_sel)
         combo.pack(pady=5)
         
         user_choice = {"idx": None, "name": None}
         def on_ok():
             selection = combo.get()
             if selection:
                 idx = int(selection.split(":")[0])
                 user_choice["idx"] = idx
                 user_choice["name"] = selection.split(":")[1].strip()
             dlg.destroy()
         tk.Button(dlg, text="OK", command=on_ok).pack(pady=10)
         root.wait_window(dlg)
         
         if user_choice["idx"] is not None:
             selected_device_idx = user_choice["idx"]
             current_mic_name = user_choice["name"]
             log_msg(f"Selected: {current_mic_name}")
             ui_refs['text_mic_name'].set_text(current_mic_name)
             if audio_stream:
                  audio_stream.stop(); audio_stream.close(); audio_stream = None
             
             sync_audio_stream('settings') 

         root.destroy()

    # [NEW] Globals for Auto-Start Sequence
    pending_rec = False
    pending_notes = ""
    session_start_ta = 0.0

    def start_actual_recording():
        global is_recording, f_gsr, writer_gsr, f_trend, writer_trend, recording_start_time
        global audio_stream, audio_buffer, audio_filename, selected_device_idx, notes_filename
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
             audio_filename = os.path.join(SESSION_DIR, "audio.wav")
             notes_filename = os.path.join(SESSION_DIR, "notes.txt")
             
             with open(notes_filename, "w") as f:
                 f.write(f"Session Notes - {ts_str}\n")
                 f.write("-" * 30 + "\n")
                 f.write(pending_notes if pending_notes else "No notes provided.")
             
             # Initialize GSR CSV
             f_gsr = open(fname_gsr, 'w', newline='')
             writer_gsr = csv.writer(f_gsr)
             writer_gsr.writerow(["Timestamp", "TA", "TA SET", "Sensitivity", "Motion"])
             
             # Initialize Trend CSV
             f_trend = open(fname_trend, 'w', newline='')
             writer_trend = csv.writer(f_trend)
             writer_trend.writerow(["Timestamp", "Calm_Score", "Delta", "Theta", "Alpha", "Beta", "Gamma"])

             audio_buffer = []
             is_recording = True 
             sync_audio_stream(current_view)
             
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
        global audio_stream, audio_buffer, audio_filename, selected_device_idx, notes_filename
        global pending_rec, pending_notes, calib_mode, calib_step, calib_phase, calib_start_time, calib_base_ta, calib_min_ta, counting_active # [FIX] Added counting_active
        
        if not is_recording:
             if selected_device_idx is None:
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
             save_audio()
             sync_audio_stream(current_view) 
             
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

    r_ts = [0.87, 0.06, 0.09, 0.04]
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
    ui_refs['btn_select_mic'].on_clicked(open_audio_select)

    r_gain = [rect_audio[0] + 0.02, rect_audio[1] + 0.01, rect_audio[2] - 0.04, 0.03]
    ax_gain = reg_ax(r_gain, settings_view_axes)
    ui_refs['slide_gain'] = Slider(ax_gain, 'Mic Gain', 1.0, 10.0, valinit=current_mic_gain, color='lime')
    ui_refs['slide_gain'].on_changed(lambda v: globals().update(current_mic_gain=v))

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
                 send_command(10, pl)
                 log_msg(f"Device Rename: {text}")
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
    system_line = ax_info.text(0.5, 0.5, "Waiting for Info...", ha="center")
    
    def teleport_off(ax_list):
        for a in ax_list: a.set_visible(False); a.set_position([1.5, 1.5, 0.01, 0.01])

    def teleport_on(ax_list):
        for a in ax_list: a.set_visible(True); a.set_position(ax_positions[a])

    teleport_off(settings_view_axes)
    teleport_on(main_view_axes)

    def req_main(e): 
        global desired_view; desired_view = 'main'
        sync_audio_stream('main')
    def req_settings(e): 
        global desired_view; desired_view = 'settings'
        sync_audio_stream('settings')
    
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
        global BASE_SENSITIVITY, saved_boost_level # [FIX] For Calibration
        global calib_mode, calib_phase, calib_step, calib_start_time, calib_base_ta, calib_min_ta, calib_vals, last_calib_ratio # [FIX] Added last_calib_ratio
        global recording_start_time, is_recording, session_start_ta # [FIX] Stats Update
        global pending_rec # [FIX] For Calibration Auto-Start
        
        global is_connected_prev # [NEW] Track prev state
        global prev_grid_state
        
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
        
        log_text.set_text("\n".join(list(log_messages))) 
        s_str = f"Name: {dev_static.get('name','?')} | SN: {dev_static.get('serial','?')} | HW: {dev_static.get('hw','?')} | FW: {dev_static.get('fw','?')} | Manuf: {dev_static.get('manuf','?')} | Batt: {device_battery_level} | Mic: {current_mic_name}"
        system_line.set_text(s_str)
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
        if audio_stream and audio_stream.active:
             raw_lvl = audio_state['peak']
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
        # But wait, 'bg_rect' is local to setup_ui in my previous edit. I need to make it global there first or put it in ui_refs.
        # EDIT: I'll assume I made it a simple local variables in setup_ui. I need to fix scope. 
        # Actually simplest to put it in ui_refs.
        
        # [NEW] Add Auto-Center Check to end of loop
        # [FIX] Removed duplicate check_auto_center() call
        
        artists = []
        if current_view == 'main':
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
                             msg = f"CALIB {calib_step}/4: SQUEEZE SENSOR\n(WAITING FOR DROP)"
                             if latest_gsr_ta < calib_base_ta - 0.02: # Trigger Drop
                                 calib_phase = 1
                                 calib_min_ta = latest_gsr_ta
                                 calib_start_time = time.time() # Reset timer for safety
                        elif calib_phase == 1: # TRACKING DROP
                             msg = f"CALIB {calib_step}/4: SQUEEZE..."
                             if latest_gsr_ta < calib_min_ta:
                                 calib_min_ta = latest_gsr_ta
                             
                             # Detect Release (75% Recovery)
                             current_drop = calib_base_ta - calib_min_ta
                             # If drop is tiny (<0.05), use fixed 0.05 rise. Else use 75% return
                             recovery_target = calib_min_ta + (current_drop * 0.75) if current_drop > 0.05 else calib_min_ta + 0.05
                             
                             if latest_gsr_ta > recovery_target:
                                 calib_phase = 3
                                 calib_start_time = time.time() # Start stability timer
                                 
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
                             
                                 # [REQ] Target: Drop = 80% of the Lower Graph Portion (0.625 * Window)
                                 # Lower Portion = 0.625 * Win. Drop = 0.8 * (0.625 * Win) = 0.5 * Win.
                                 # Win = Drop / 0.5 = Drop * 2.0
                                 measured_win = total_drop / 0.5
                                 measured_win = max(0.05, min(15.0, measured_win))
                                 
                                 # [REQ] Normalize for Auto-Boost
                                 # If Auto-Boost will be restored, EffectiveWin = BASE * Factor
                                 # We want EffectiveWin == measured_win
                                 # So BASE = measured_win / Factor
                                 factor = 1.0
                                 if saved_boost_level > 0:
                                     mult = [1.0, 0.6, 1.0, 1.4][saved_boost_level]
                                     safe_ta = max(1.0, calib_base_ta) # Use current TA for normalization
                                     factor = math.pow((safe_ta / 2.0), mult)
                                     new_base = measured_win / factor
                                 else:
                                     new_base = measured_win
                                 
                                 BASE_SENSITIVITY = new_base if calib_step == 1 else max(BASE_SENSITIVITY, new_base)
                                 log_msg(f"Calib {calib_step}: Drop={total_drop:.2f}, Win={measured_win:.2f}, Fac={factor:.2f} -> Base={new_base:.2f}")
                                 # txt_win_val.set_text(f"{1.0/measured_win:.2f}") # Display Effective Sensitivity
                             
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
                            msg = "CALIB 4/4: DEEP BREATH\n(WAITING FOR DROP)"
                            if latest_gsr_ta < calib_base_ta - 0.02:
                                calib_phase = 1
                                calib_min_ta = latest_gsr_ta
                                calib_start_time = time.time() # [REQ] Start Stability Timer
                        elif calib_phase == 1:
                            msg = "CALIB 4/4: EXHALE / RELEASE..."
                            
                            # Track Minimum (Deepest Point)
                            if latest_gsr_ta < calib_min_ta:
                                calib_min_ta = latest_gsr_ta
                                calib_start_time = time.time() # Reset Stability Timer on new low
                            
                            # Check Stability (No new low for 1.5s)
                            if time.time() - calib_start_time > 1.5:
                                # Validate Drop
                                total_drop = calib_base_ta - calib_min_ta
                                ratio = total_drop / BASE_SENSITIVITY
                                last_calib_ratio = ratio # [DEBUG]
                                
                                # [FIX] Relax Lower Bound (0.50 -> 0.10) to allow weak breaths
                                if ratio < 0.10 or ratio > 0.90:
                                    log_msg(f"Breath Calib Failed: Ratio {ratio:.2f} (Target 0.1-0.9)")
                                    # Go to Error Phase
                                    calib_phase = 5
                                    calib_start_time = time.time()
                                else:
                                    # Success
                                    calib_phase = 2
                                    calib_start_time = time.time()
                                    update_gsr_center(latest_gsr_ta)
                                
                        elif calib_phase == 2:
                            msg = "CALIB: COMPLETE!"
                            if time.time() - calib_start_time > 2.0: # Show Success for 2s
                                # [REQ] Restore previous Auto-Boost Level
                                log_msg(f"Restoring Boost: {saved_boost_level}")
                                set_boost(saved_boost_level)
                                
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
                                    if ovl: ovl.set_text("")
                                    log_msg("Calibration Complete")

                        elif calib_phase == 3:
                             msg = "SESSION STARTED"
                             # [REQ] Start TA Counter Here (After Success)
                             if not counting_active: toggle_count(None)
                             
                             if time.time() - calib_start_time > 2.0:
                                  calib_mode = False
                                  if ovl: ovl.set_text("")
                                  log_msg("Calibration Sequence Finished")

                        elif calib_phase == 5:
                             # [NEW] Error Phase
                             msg = f"CALIBRATION FAILED\nRATIO: {last_calib_ratio:.2f} (0.1-0.9)"
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
             sync_audio_stream('main') 
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
        t.join(timeout=2.0)
        print("Shutdown Complete.")
