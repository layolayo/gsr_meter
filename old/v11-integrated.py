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
import json # [NEW] Config Persistence
import sounddevice as sd
import scipy.io.wavfile as wav
try:
    import hid
except ImportError:
    hid = None # Handle missing library gracefully


# --- CONFIGURATION ---
FS = 250 # Hertz of headset datastream
MAX_WINDOW_SEC = 10
BUFFER_SIZE = int(FS * MAX_WINDOW_SEC)
HISTORY_LEN = 2500 
WARMUP_SAMPLES = FS * 3
SCORE_SMOOTHING_WINDOW = int(FS * 2.0) 

# --- SPIKE DETECTION SETTINGS ---
SPIKE_SIGMA_MULTIPLIER = 7.0 
SPIKE_MIN_THRESHOLD = 1500  
SPIKE_MAX_THRESHOLD = 8000
SPIKE_DISPLAY_SECONDS = 1.0  

# File Naming
FILENAME_MAIN = "brainwave_session"
FILENAME_DETAILED = "brainwave_detailed"
CONFIG_FILE = "config.json" # [NEW] Settings File

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
bands_history = {k: collections.deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN) for k in
                 ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']}
log_messages = collections.deque(["System Ready"], maxlen=10)
is_connected = False
current_state = "LINK : DISCONNECTED" # ... (Globals)
current_theme = "Light" # [NEW]
headset_on_head = False
app_running = True
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
ADVERTISED_NAME = "Unknown"

# Biofeedback
current_calm_score = 0
current_focus_score = 0
calm_history = collections.deque(maxlen=SCORE_SMOOTHING_WINDOW)
focus_history = collections.deque(maxlen=SCORE_SMOOTHING_WINDOW)
use_colour_active = False
light_active = True 
biofeedback_source = 'Calm' 

# Logic Settings
current_smoothing = 0.1
event_detected = False
spike_detection_enabled = False

# Spike Detection State
last_spike_time = 0
current_spike_mag = 0
spike_detected_flag = False
last_on_signal_time = 0

# Control Defaults
current_window_sec = 0.5
baseline_window_sec = 2.0
coincidence_window = 0.5
global_percent = 20
triggers_enabled = False

# Volatility Settings
volatility_window = 25  
max_vol_scale = 2000.0  

# Trigger Config
triggers = {
    'Delta': {'mode': 1, 'last_seen': 0, 'active': False, 'dynamic_thresh': 0},
    'Theta': {'mode': 1, 'last_seen': 0, 'active': True, 'dynamic_thresh': 0},
    'Alpha': {'mode': -1, 'last_seen': 0, 'active': True, 'dynamic_thresh': 0},
    'Beta': {'mode': 1, 'last_seen': 0, 'active': True, 'dynamic_thresh': 0},
    'Gamma': {'mode': 1, 'last_seen': 0, 'active': False, 'dynamic_thresh': 0}
}
colors = {'Delta': 'blue', 'Theta': 'green', 'Alpha': 'orange', 'Beta': 'red', 'Gamma': 'purple'}

# CSV Handles
csv_file = None
csv_writer = None
csv_file_detailed = None
csv_writer_detailed = None
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
audio_state = {'peak': 0.0, 'debug_ctr': 0} # [NEW] Mutable State Container
# current_audio_level removed in favor of audio_state

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

# --- GSR STATE & SETTINGS ---
GSR_VENDOR_ID = 0x1fc9
GSR_PRODUCT_ID = 0x0003
GSR_V_SOURCE = 6.371
GSR_R_REF = 83.0
gsr_thread = None
gsr_connected = False
gsr_current_ta = 0.0
gsr_graph_data = collections.deque([0] * 500, maxlen=500)
gsr_center_ta = 2.0
gsr_base_sensitivity = 0.20
gsr_booster_level = 0 # 0=Off, 1=Lo, 2=Med, 3=Hi
gsr_counting_active = False
gsr_ta_total = 0.0
gsr_last_ta_frame = 0.0
gsr_is_artifact = False
gsr_bg_img = None
gsr_dial_img = None

class GSRReader(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        self.connected = False

    def run(self):
        global gsr_connected, gsr_current_ta
        
        if not hid:
            log_msg("GSR: HID lib missing")
            return
            
        while self.running:
            try:
                # Poll for device if not connected
                if not self.connected:
                    try:
                        self.h = hid.device()
                        self.h.open(GSR_VENDOR_ID, GSR_PRODUCT_ID)
                        self.h.set_nonblocking(0) # Blocking read
                        self.connected = True
                        log_msg("GSR: Device Connected")
                        gsr_connected = True
                    except:
                        time.sleep(1.0)
                        continue

                # Read Data
                try:
                    data = self.h.read(64, timeout_ms=20)
                except:
                    self.connected = False
                    gsr_connected = False
                    continue

                if data and len(data) >= 4 and data[0] == 0x01:
                    raw_val = (data[2] << 8) | data[3]
                    voltage = raw_val / 10000.0
                    if voltage >= (GSR_V_SOURCE - 0.005): ohms = 999999.9
                    else:
                        try: ohms = (voltage * GSR_R_REF) / (GSR_V_SOURCE - voltage)
                        except: ohms = 999999.9
                    try:
                        ta = (ohms * 1000 / (ohms * 1000 + 21250)) * 5.559 + 0.941
                    except: ta = 0.0
                    
                    gsr_current_ta = ta
                elif not data:
                    pass
            except Exception as e:
                self.connected = False
                gsr_connected = False
                time.sleep(1.0)

    def stop(self):
        self.running = False

# --- FUNCTIONS ---
def calculate_focus_score(bands):
    try:
        theta = bands[1]
        beta = max(1e-6, bands[3])
        if theta < 0.1 and beta < 0.1: return 0
        ratio = beta / max(1e-6, theta)
        raw_score = 50 + 50 * math.log10(ratio)
        return int(np.clip(raw_score, 0, 100))
    except: return 0

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
    global headset_on_head, current_state, is_recording, current_calm_score, current_focus_score
    global samples_since_contact, last_packet_time, zi_ch1, eeg_buffer
    global last_spike_time, current_spike_mag, spike_detected_flag, current_client, timestamp_queue
    global last_on_signal_time, current_window_sec, total_samples_recorded

    last_packet_time = time.time()
    if not data.startswith(b'CMSN'): return
    if timestamp_queue is not None:
        try: timestamp_queue.put_nowait(data)
        except: pass

    body = data[6:]
    def parse_payload(payload):
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
            elif key == 0x08: idx += 1
            elif key == 0x10: idx += 1
            elif key == 0x01: pass
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
        new_raw = parse_payload(body)
        if len(new_raw) > 0:
            filt_chunk, zi_ch1 = lfilter(b_filt, a_filt, new_raw, zi=zi_ch1)
            current_state = "LINK : STREAMING"
            detailed_rows = []
            for i, val in enumerate(filt_chunk):
                eeg_buffer.append(val) 
                if headset_on_head: samples_since_contact += 1
                else: samples_since_contact = 0
                is_warmed_up = samples_since_contact > WARMUP_SAMPLES
                req_samples = int(FS * current_window_sec)
                bands = [0]*5; calm_inst=0; focus_inst=0
                if headset_on_head and is_warmed_up and len(eeg_buffer) >= req_samples:
                     bands = calculate_relative_bands(list(eeg_buffer)[-req_samples:])
                     try:
                         raw_c = 30 * math.log10(bands[1] / (bands[2] + max(1e-6, bands[3]))) + 50
                         calm_inst = int(np.clip(raw_c, 0, 100))
                         focus_inst = calculate_focus_score(bands)
                     except: pass
                     calm_history.append(calm_inst)
                     focus_history.append(focus_inst)
                     if i == len(filt_chunk) - 1:
                         global current_calm_score, current_focus_score
                         if len(calm_history) > 0: current_calm_score = int(np.mean(calm_history))
                         else: current_calm_score = calm_inst
                         if len(focus_history) > 0: current_focus_score = int(np.mean(focus_history))
                         else: current_focus_score = focus_inst
                    
                     for k, b_val in zip(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], bands):
                         prev = bands_history[k][-1] if len(bands_history[k]) > 0 else 0
                         alpha = 1.0 - current_smoothing
                         alpha = max(0.01, min(1.0, alpha))
                         sm_val = (prev * (1.0 - alpha)) + (b_val * alpha)
                         bands_history[k].append(sm_val)
                if is_recording and csv_writer_detailed and recording_start_time:
                     sample_idx = total_samples_recorded + i
                     rel_ts_val = sample_idx / FS
                     abs_ts = (recording_start_time + timedelta(seconds=rel_ts_val)).strftime('%H:%M:%S.%f')
                     rel_ts = f"{rel_ts_val:.4f}"
                     detailed_rows.append([
                         abs_ts, rel_ts, new_raw[i], int(headset_on_head),
                         bands[0], bands[1], bands[2], bands[3], bands[4],
                         current_calm_score, current_focus_score 
                     ])

            if detailed_rows:
                try: 
                    csv_writer_detailed.writerows(detailed_rows)
                    total_samples_recorded += len(detailed_rows)
                except: pass

            if is_recording and csv_writer:
                ts = datetime.now().strftime('%H:%M:%S.%f')
                row = [ts, current_state, int(headset_on_head), 1 if event_detected else 0, 1 if spike_detected_flag else 0, current_focus_score, current_calm_score] + bands + list(new_raw[:50])
                csv_writer.writerow(row)

            if spike_detection_enabled and headset_on_head and len(eeg_buffer) > 250:
                 recent = list(eeg_buffer)[-250:]
                 diffs = np.diff(recent)
                 med_val = np.median(diffs)
                 mad = np.median(np.abs(diffs - med_val))
                 thresh = max(mad * 1.4826 * SPIKE_SIGMA_MULTIPLIER, SPIKE_MIN_THRESHOLD)
                 new_diffs = diffs[-len(filt_chunk):]
                 max_jump = np.max(np.abs(new_diffs))
                 if thresh < max_jump < SPIKE_MAX_THRESHOLD:
                     last_spike_time = time.time(); current_spike_mag = int(max_jump); spike_detected_flag = True
                 else: spike_detected_flag = False
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
        cfg = {
            'mic_name': current_mic_name,
            'mic_gain': current_mic_gain,
            'smoothing': current_smoothing,
            'window_sec': current_window_sec,
            'vol_thresh': max_vol_scale,
            'global_percent': global_percent,
            'light_active': light_active, # [NEW]
            'use_colour': use_colour_active, # [NEW]
            'src_mode': biofeedback_source,
            'bt_address': DEVICE_ADDRESS,
            'bt_name': ADVERTISED_NAME, 
            'triggers': {},
            # [NEW] Persist Trigger Settings
            'coin_win': coincidence_window,
            'base_win': baseline_window_sec,
            'master_active': triggers_enabled,
            'coin_win': coincidence_window,
            'base_win': baseline_window_sec,
            'master_active': triggers_enabled,
            'spike_active': spike_detection_enabled,
            # [NEW] GUI Settings
            'gui_theme': current_theme
        }
        for k, v in triggers.items():
            cfg['triggers'][k] = {'active': v['active'], 'mode': v['mode']}
            
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=4)
        print(f"[Config] Saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"[Config] Save Error: {e}")

def load_config():
    global current_mic_name, current_mic_gain, current_smoothing, current_window_sec
    global max_vol_scale, global_percent, triggers
    global light_active, use_colour_active, biofeedback_source
    global DEVICE_ADDRESS, ADVERTISED_NAME # [NEW]
    
    if not os.path.exists(CONFIG_FILE):
        print("[Config] No file found. Using defaults.")
        return

    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            
        current_mic_name = cfg.get('mic_name', "Default")
        current_mic_gain = float(cfg.get('mic_gain', 3.0))
        current_smoothing = float(cfg.get('smoothing', 0.1))
        current_window_sec = float(cfg.get('window_sec', 0.5))
        max_vol_scale = float(cfg.get('vol_thresh', 2000.0))
        global_percent = int(cfg.get('global_percent', 20))
        
        light_active = cfg.get('light_active', True)
        use_colour_active = cfg.get('use_colour', False)
        light_active = cfg.get('light_active', True)
        use_colour_active = cfg.get('use_colour', False)
        biofeedback_source = cfg.get('src_mode', 'Calm')
        
        # [NEW] Restore Theme
        global current_theme
        current_theme = cfg.get('gui_theme', 'Light')
        
        # [NEW] Restore Bluetooth
        saved_addr = cfg.get('bt_address', None)
        saved_name = cfg.get('bt_name', "Unknown")
        if saved_addr:
             DEVICE_ADDRESS = saved_addr
             ADVERTISED_NAME = saved_name
             print(f"[Config] Restored Target Device: {ADVERTISED_NAME} ({DEVICE_ADDRESS})")
        
        # [NEW] Restore Trigger Settings
        global coincidence_window, baseline_window_sec, triggers_enabled, spike_detection_enabled
        coincidence_window = float(cfg.get('coin_win', 0.5))
        baseline_window_sec = float(cfg.get('base_win', 0.5))
        triggers_enabled = cfg.get('master_active', True)
        spike_detection_enabled = cfg.get('spike_active', False)

        t_cfg = cfg.get('triggers', {})
        for k, v in t_cfg.items():
            if k in triggers:
                triggers[k]['active'] = v.get('active', True)
                triggers[k]['mode'] = v.get('mode', 1)
        
        print(f"[Config] Loaded settings. Mic: {current_mic_name}, Gain: {current_mic_gain}")
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
    global is_connected, current_state, last_packet_time, device_battery_level, current_client, timestamp_queue, REAL_SERIAL_STR, DEVICE_ADDRESS, ADVERTISED_NAME
    STRATEGIES = [{"seq": 2, "type": 2, "delay": 1.0}, {"seq": 50, "type": 2, "delay": 1.0}, {"seq": 1, "type": 1, "delay": 2.0}]
    
    while True:
        try:
            # [FIX] Always Scan (Removes Fast Connect Loop)
            # if DEVICE_ADDRESS and ADVERTISED_NAME != "Unknown":
            #      log_msg(f"Fast Connect: {ADVERTISED_NAME} ({DEVICE_ADDRESS})...")
            
            # Reset address if we need to scan? No, keep it as 'target'.
            # Actually, to force a refresh, we should ignore the old address object and find a new one.
            
            log_msg(f"[SCAN] Scanning for {ADVERTISED_NAME if ADVERTISED_NAME else 'Devices'}...")
            devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
            candidate = None
            
            # [NEW] Prioritize Saved Name
            if ADVERTISED_NAME and ADVERTISED_NAME != "Unknown":
                for d, adv in devices.values():
                    # Strict Check on Name
                    if d.name == ADVERTISED_NAME:
                        candidate = d; break
            
            # [NEW] Fallback Search
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
                log_msg("Device Not Found. Retrying...")
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
                
                try:
                    s_bytes = await client.read_gatt_char(UUID_SERIAL)
                    REAL_SERIAL_STR = s_bytes.decode('utf-8').strip()
                except: raise Exception("Serial Fail")

                dev_info = {'name': ADVERTISED_NAME, 'serial': REAL_SERIAL_STR}
                try: dev_info['manuf'] = (await client.read_gatt_char(UUID_MANUFACTURER)).decode()
                except: dev_info['manuf'] = '?'
                try: dev_info['fw'] = (await client.read_gatt_char(UUID_FIRMWARE)).decode()
                except: dev_info['fw'] = '?'
                try: dev_info['hw'] = (await client.read_gatt_char(UUID_HARDWARE)).decode()
                except: dev_info['hw'] = '?'
                ui_update_queue.put(dev_info)

                current_client = client; timestamp_queue = asyncio.Queue()
                listen_task = asyncio.create_task(listener(client, timestamp_queue))

                handshake_done = False
                for strat in STRATEGIES:
                    try:
                        await safe_write(client, create_auth_packet(strat['seq'], strat['type']), delay=strat['delay'])
                        if not client.is_connected:
                            listen_task.cancel(); await client.disconnect(); await asyncio.sleep(1.0); await client.connect()
                            current_client = client; timestamp_queue = asyncio.Queue(); listen_task = asyncio.create_task(listener(client, timestamp_queue)); continue
                        await safe_write(client, CMD_2_CONFIG, delay=0.2)
                        await client.start_notify(UUID_NOTIFY, notification_handler)
                        await safe_write(client, CMD_3_STREAM, delay=0.2)
                        await safe_write(client, CMD_4_SETUP, delay=0.2)
                        await safe_write(client, CMD_5_FINAL, delay=0.2)
                        handshake_done = True; break
                    except: pass
                
                if not handshake_done: raise Exception("Handshake Failed")

                log_msg("Streaming...")
                is_connected = True; current_state = "LINK : STREAMING"
                
                # [NEW] Initial Battery Check
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
                # If Fast Connect failed, clear address so we scan next time
                if "Conn Fail" in str(e) or "Serial" in str(e):
                     DEVICE_ADDRESS = None 
            finally:
                if 'client' in locals() and client.is_connected: await client.disconnect()
                if 'listen_task' in locals(): listen_task.cancel()
            if not is_connected: await asyncio.sleep(2.0)
        except: await asyncio.sleep(2.0)

def run_ble():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bluetooth_task())

if __name__ == "__main__":
    t = threading.Thread(target=run_ble, daemon=True)
    t.start()
    
    # [NEW] Start GSR Thread
    if hid: 
        gsr_thread = GSRReader()
        gsr_thread.start()
    else: print("[GSR] HID Library not found -- GSR View will be simulated/empty.")

    load_config() # [NEW] Load Settings First
    # [MOVED] sync_audio_stream call moved to END of init to ensure GUI exists
    pass
    
    # --- GUI ---
    # --- GUIDE ---
    # Fig: 14x9 inches.
    # We want extra top padding.
    # Current Top: 0.95. Let's push Axes down.
    fig = plt.figure(figsize=(14, 9))
    try: fig.canvas.manager.set_window_title("Emergent Knowledge Brainwave Scanner")
    except: pass # Some backends might not support
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    current_view = 'main'
    desired_view = 'main'
    ax_positions = {}
    main_view_axes = []
    settings_view_axes = []
    gsr_view_axes = [] # [NEW]
    
    def get_color_pkt(score, mode):
        # ... (Unchanged)
        r, g, b = 0, 0, 0
        if score < 50:
            ratio = score / 50.0
            r = 15; b = 0
            g = int(7 * ratio)
        else:
            ratio = (score - 50) / 50.0
            g = 7; b = 0
            r = int(15 * (1 - ratio))
        r = max(0, min(15, r)); g = max(0, min(7, g)); b = max(0, min(3, b))
        pl = bytearray([0x10, 255, 0x80, 0x80|b, 0x80|g, 0x00|r])
        return pl

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

    # --- MAIN GRAPH ---
    # Was [0.05, 0.20, 0.90, 0.75]. Height 0.75 -> Top 0.95 (Too tight)
    # Then [0.05, 0.20, 0.90, 0.70]. Height 0.70 -> Top 0.90 (Too much space)
    # Now [0.05, 0.20, 0.90, 0.73]. Height 0.73 -> Top 0.93 (Just right?)
    ax = reg_ax([0.05, 0.20, 0.90, 0.73], main_view_axes)
    
    ax_vol = reg_ax([0.96, 0.20, 0.02, 0.73], main_view_axes)
    ax_vol.set_xticks([]); ax_vol.yaxis.tick_right(); ax_vol.set_ylim(0, max_vol_scale)
    ax_vol.set_title("Vol", color='#333333', fontsize=8)
    vol_bar = ax_vol.bar(["V"], [0], color='cyan', alpha=0.6)
    
    lines = {b: ax.plot([], [], lw=2, label=b, color=colors[b])[0] for b in bands_history}
    mean_lines = {b: ax.axhline(y=0, color=colors[b], linestyle='--', alpha=0.3) for b in bands_history}
    thresh_lines = {b: ax.axhline(y=0, color=colors[b], linestyle=':', alpha=0.8) for b in bands_history}
    
    ax.legend(loc="upper left")
    for k in bands_history:
        v = triggers[k]['active']
        lines[k].set_visible(v)
        mean_lines[k].set_visible(v)
        thresh_lines[k].set_visible(v)

    ax.set_xlim(0, HISTORY_LEN); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3) 
    ax.set_title(f"Brainwave Event Detector (24-bit @ {FS}Hz)")
    
    txt_conn = ax.text(0.15, 1.05, "LINK : DISCONNECTED", transform=ax.transAxes, color='gray', fontweight='bold')
    txt_wear = ax.text(0.40, 1.05, "HEADSET : --", transform=ax.transAxes, color='gray', fontweight='bold')
    txt_batt = ax.text(0.60, 1.05, "BATT: --", transform=ax.transAxes, color='gray', fontweight='bold')
    # [NEW] Audio Status Indicator
    # [NEW] Audio Status Indicator - Start as NO MIC
    txt_audio = ax.text(0.75, 1.05, "AUDIO: NO MIC", transform=ax.transAxes, color='gray', fontweight='bold', fontsize=9)
    txt_score = ax.text(0.50, 0.95, "Score: --", transform=ax.transAxes, ha="center")
    
    rec_text = ax.text(0.95, 0.95, "● REC", transform=ax.transAxes, color='red', visible=False, fontweight='bold') 
    spike_text = ax.text(0.5, 0.65, "⚡ SPIKE DETECTED", transform=ax.transAxes, 
                         ha="center", va="center", fontsize=24, fontweight='bold', color='red',
                         bbox=dict(facecolor='yellow', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.5'),
                         visible=False)
    event_text = ax.text(0.5, 0.5, "⚠️ EVENT DETECTED", transform=ax.transAxes,
                         ha="center", va="center", fontsize=30, fontweight='bold', color='red',
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'), 
                         visible=False)
    
    ui_refs = {}

    start_x = 0.05; col_width = 0.19; y_pos = 0.12
    for i, band in enumerate(bands_history.keys()):
        x_pos = start_x + (i * col_width)
        r_c = [x_pos, y_pos, 0.10, 0.04]
        ax_chk = reg_ax(r_c, main_view_axes)
        try:
            init_state = triggers[band]['active']
            chk = CheckButtons(ax_chk, [band], [init_state],
                               label_props={'color': [colors[band]], 'fontweight': ['bold']})
            if hasattr(chk, 'rectangles') and len(chk.rectangles) > 0:
                chk.rectangles[0].set_facecolor(colors[band])
        except: chk = CheckButtons(ax_chk, [band], [True])
        ui_refs[f'chk_{band}'] = chk

        r_d = [x_pos + 0.11, y_pos, 0.04, 0.04]
        ax_dir = reg_ax(r_d, main_view_axes)
        btn_dir = Button(ax_dir, "↑" if triggers[band]['mode'] == 1 else "↓", color='white', hovercolor='0.9')
        ui_refs[f'btn_{band}'] = btn_dir

        def make_toggle_vis(b, c):
             def toggle(label):
                  state = c.get_status()[0]
                  triggers[b]['active'] = state
                  lines[b].set_visible(state)
                  mean_lines[b].set_visible(state)
                  thresh_lines[b].set_visible(state)
                  plt.draw()
             return toggle
        def make_toggle_dir(b, btn):
             def toggle_dir(event):
                 triggers[b]['mode'] *= -1
                 btn.label.set_text("↑" if triggers[b]['mode'] == 1 else "↓")
                 plt.draw()
             return toggle_dir

        chk.on_clicked(make_toggle_vis(band, chk))
        btn_dir.on_clicked(make_toggle_dir(band, btn_dir))

    r_rl = [0.05, 0.06, 0.12, 0.04]
    ax_relock = reg_ax(r_rl, main_view_axes)
    ui_refs['btn_relock'] = Button(ax_relock, 'Re-Calib', color='lightblue')
    ui_refs['btn_relock'].on_clicked(lambda e: eeg_buffer.clear())

    r_rc = [0.20, 0.06, 0.12, 0.04]
    ax_rec = reg_ax(r_rc, main_view_axes)
    ui_refs['btn_rec'] = Button(ax_rec, 'Record', color='lightgreen')
    
    import tkinter as tk
    from tkinter import simpledialog, ttk

    # --- AUDIO STREAM LOGIC (SMART STREAM) ---
    # --- AUDIO STREAM LOGIC (SMART STREAM) ---
    # [NEW] Global Callback for reliable state access
    def global_audio_callback(indata, frames, time, status):
        global audio_buffer, audio_state, current_mic_gain, is_recording
        
        # 1. Calculate Level (PEAK)
        gained_sig = indata * current_mic_gain
        peak = np.max(np.abs(gained_sig))
        audio_state['peak'] = peak # Update Mutable Dict
        
        # RECORD
        if is_recording:
             amplified = np.clip(gained_sig, -1.0, 1.0)
             audio_buffer.append(amplified.copy())

    def sync_audio_stream(target_view):
        global audio_stream, selected_device_idx, current_mic_gain, is_recording, audio_state, current_mic_name
        
        should_be_on = is_recording or (target_view == 'settings')

        # [NEW] Phase 1: Always Resolve Device Index (Even if Stream is OFF)
        # This ensures 'selected_device_idx' is set for when user hits Record later
        if selected_device_idx is None:
            try:
                devs = sd.query_devices()
                target_mic = current_mic_name
                
                # [STRICT FLOW] 1. Reset State
                selected_device_idx = None
                current_mic_name = "NO MIC" 
                
                # [STRICT FLOW] 2. Check Target (Preferred)
                # [NEW] Log all devices for debug
                for i, d in enumerate(devs):
                    if d['max_input_channels'] > 0: log_msg(f"Dev {i}: {d['name']}")

                if target_mic and target_mic != "Default" and target_mic != "N/A" and target_mic != "NO MIC":
                     for i, d in enumerate(devs):
                         # [NEW] Fuzzy Match: Check if target (e.g. "TONOR") is IN the device name
                         if d['max_input_channels'] > 0 and (target_mic in d['name'] or d['name'] in target_mic):
                             # STRICT PROBE: Read Samples
                             try:
                                 with sd.InputStream(device=i, samplerate=None, channels=1) as s:
                                     s.read(10) # Must actually read data
                                 
                                 selected_device_idx = i
                                 current_mic_name = d['name'] # Update to exact hardware name
                                 log_msg(f"Mic Verified (Target): {current_mic_name}")
                                 break
                             except Exception as pe:
                                 log_msg(f"Probe Fail (Target): {d['name']} ({pe})")
                
                # [STRICT FLOW] 3. Fallback (If Target Failed/None)
                if selected_device_idx is None:
                    # Only fallback if we don't have a valid target or target failed
                    for i, d in enumerate(devs):
                        if d['max_input_channels'] > 0:
                             # STRICT PROBE: Read Samples
                             try:
                                 with sd.InputStream(device=i, samplerate=None, channels=1) as s:
                                     s.read(10)
                                 
                                 selected_device_idx = i
                                 current_mic_name = d['name']
                                 log_msg(f"Mic Verified (Fallback): {current_mic_name}")
                                 break
                             except: pass

                # [STRICT FLOW] 4. Update UI (Final Result)
                # Regardless of what happened, update UI to match current_mic_name
                # (which is either Verified Name or "NO MIC")
                if 'text_mic_name' in ui_refs:
                     ui_refs['text_mic_name'].set_text(current_mic_name)
                if 'txt_audio' in locals() or 'txt_audio' in globals(): # Safety
                     txt_audio.set_text(f"AUDIO: {current_mic_name[:15]}")
                     
            except Exception as e: log_msg(f"Dev Check Err: {e}")

        # [NEW] Phase 2: Stream Activation Logic
        
        # OFF Transition
        if not should_be_on:
            if audio_stream:
                audio_stream.stop(); audio_stream.close(); audio_stream = None
                log_msg("Audio Stream: OFF")
            return

        # ON Transition / Maintenance
        if audio_stream and audio_stream.active:
             return # Already on
             
        # Needs to be ON
        try:
             # Device Resolution is now handled above ^^^
             # Just safety check
            if selected_device_idx is None: return 
            
            # Fallback Logic for Sample Rate
            rates_to_try = [44100, 48000, None] # None = Device Default
            stream_created = False
            
            # Ensure previous stream is really closed
            if audio_stream:
                 try: audio_stream.close()
                 except: pass
                 audio_stream = None
            time.sleep(0.2) # Yield to OS to free device
            
            for sr in rates_to_try:
                try:
                    log_msg(f"Trying SR: {sr if sr else 'Auto'}...")
                    audio_stream = sd.InputStream(
                        samplerate=sr, channels=1, device=selected_device_idx, 
                        callback=global_audio_callback, blocksize=1024
                    )
                    audio_stream.start()
                    stream_created = True
                    log_msg(f"Audio Stream: ON ({sr if sr else 'Auto'} Hz)")
                    break
                except Exception as e:
                    log_msg(f"SR {sr} Fail: {e}")
                    if audio_stream: 
                        try: audio_stream.close()
                        except: pass; 
                        audio_stream = None
                    time.sleep(0.3) # Wait before retry
            
            if not stream_created:
                raise Exception("All Sample Rates Failed")

        except Exception as e: log_msg(f"Stream Err: {e}")

    # [NEW] Audio Selection Dialog (Independent)
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
         
         # Try to find current
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
             # FORCE RESTART STREAM (Even if previously crashed/None)
             if audio_stream:
                  audio_stream.stop(); audio_stream.close(); audio_stream = None
             
             sync_audio_stream('settings') # Force Restart with new device

         root.destroy()

    def toggle_rec(e):
        global is_recording, csv_file, csv_writer, csv_file_detailed, csv_writer_detailed, recording_start_time
        global audio_stream, audio_buffer, audio_filename, total_samples_recorded, selected_device_idx
        
        if not is_recording:
             # Just checks
             if selected_device_idx is None:
                 log_msg("Err: No Mic Selected!")
                 return
                 
             root = tk.Tk(); root.withdraw()
             # [NEW] Enhanced Notes Dialog
             note_data = {"text": None}
             
             dlg = tk.Toplevel(root)
             dlg.title("Session Notes")
             dlg.geometry("500x400")
             
             tk.Label(dlg, text="Enter Session Details:", font=("Arial", 10, "bold")).pack(pady=5)
             
             # Template
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
             
             notes = note_data["text"]
             
             # [NEW] Organize Data
             DATA_DIR = "Session_Data"
             os.makedirs(DATA_DIR, exist_ok=True) # Ensure folder exists
             
             ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
             # [NEW] Prepend Directory
             fname_main = os.path.join(DATA_DIR, f"{FILENAME_MAIN}_{ts_str}.csv")
             fname_detail = os.path.join(DATA_DIR, f"{FILENAME_DETAILED}_{ts_str}.csv")
             audio_filename = os.path.join(DATA_DIR, f"brainwave_audio_{ts_str}.wav")
             notes_filename = os.path.join(DATA_DIR, f"brainwave_notes_{ts_str}.txt")
             
             try:
                 with open(notes_filename, "w") as f:
                     f.write(f"Session Notes - {ts_str}\n")
                     f.write("-" * 30 + "\n")
                     f.write(notes if notes else "No notes provided.")
                 
                 csv_file = open(fname_main, 'w', newline='')
                 csv_writer = csv.writer(csv_file)
                 header = ["Timestamp", "State", "Headset_On", "Event_Detected", "Spike_Detected", "Focus_Score",
                           "Calm_Score", "Delta", "Theta",
                           "Alpha", "Beta", "Gamma"] + [f"Raw_{i}" for i in range(50)]
                 csv_writer.writerow(header)

                 csv_file_detailed = open(fname_detail, 'w', newline='')
                 csv_writer_detailed = csv.writer(csv_file_detailed)
                 header_detail = ["Abs_Time", "Rel_Time", "Raw_24bit", "Headset", "Delta", "Theta", "Alpha",
                                  "Beta", "Gamma", "Calm_Score", "Focus_Score"]
                 csv_writer_detailed.writerow(header_detail)

                 audio_buffer = []
                  # Start Stream logic handled by sync_audio_stream
                 is_recording = True 
                 sync_audio_stream(current_view)
                 
                 recording_start_time = datetime.now()
                 total_samples_recorded = 0
                 
                 ui_refs['btn_rec'].label.set_text("Stop")
                 ui_refs['btn_rec'].color = 'salmon'
                 rec_text.set_visible(True)
                 ax.set_title(f"Recording: {ts_str}")
                 log_msg(f"Started: {ts_str}")
             except Exception as ex: log_msg(f"Error: {ex}")
        else:
             is_recording = False
             if csv_file: csv_file.close()
             if csv_file_detailed: csv_file_detailed.close()
             save_audio()
             sync_audio_stream(current_view) # Re-eval if we need stream
             
             ui_refs['btn_rec'].label.set_text("Record")
             ui_refs['btn_rec'].color = 'lightgreen'
             rec_text.set_visible(False)
             ax.set_title(f"Brainwave Event Detector (24-bit @ {FS}Hz)")
             log_msg("Recording Saved")

    ui_refs['btn_rec'].on_clicked(toggle_rec)

    r_ts = [0.80, 0.06, 0.15, 0.04]
    ax_to_set = reg_ax(r_ts, main_view_axes)
    ui_refs['btn_to_settings'] = Button(ax_to_set, "Settings >", color='lightgray')

    # --- SETTINGS PAGE ELEMENTS ---
    rect_trig = [0.05, 0.05, 0.28, 0.18]
    rect_gui  = [0.05, 0.28, 0.28, 0.25] # [NEW] Top Left
    rect_proc = [0.35, 0.05, 0.28, 0.18]
    rect_audio = [0.35, 0.28, 0.28, 0.25] 
    rect_conf = [0.65, 0.28, 0.33, 0.25] 
    rect_log = [0.65, 0.05, 0.33, 0.20] 
    
    ax_trig_bg = create_panel_ax(rect_trig, "Trigger Settings")
    ax_gui_bg  = create_panel_ax(rect_gui, "GUI Settings") # [NEW]
    ax_proc_bg = create_panel_ax(rect_proc, "Signal Processing")
    ax_conf_bg = create_panel_ax(rect_conf, "Device Controls")
    ax_audio_bg = create_panel_ax(rect_audio, "Audio Input Control")
    
    # 0. GUI Settings (Theme)
    r_thm = [rect_gui[0] + 0.02, rect_gui[1] + 0.14, 0.10, 0.08]
    ax_thm = reg_ax(r_thm, settings_view_axes)
    ax_thm.set_facecolor('#e0e0e0'); ax_thm.axis('off')
    
    # Init Check
    is_dark = (current_theme == 'Dark')
    ui_refs['rad_theme'] = RadioButtons(ax_thm, ['Light', 'Dark'], active=(1 if is_dark else 0))
    
    def apply_theme(theme_name):
        global current_theme
        current_theme = theme_name
        
        # DEFINE PALETTES
        if theme_name == 'Dark':
            c_bg = '#1e1e1e'; c_fg = 'white'; c_panel = '#383838'; c_grid = '#555555'; c_ax = '#2b2b2b'
        else: # Light
            c_bg = '#f0f0f0'; c_fg = 'black'; c_panel = '#e0e0e0'; c_grid = 'gray'; c_ax = 'white'
            
        fig.patch.set_facecolor(c_bg)
        
        # Apply to Main Graph
        ax.set_facecolor(c_ax)
        ax.tick_params(colors=c_fg)
        ax.xaxis.label.set_color(c_fg); ax.yaxis.label.set_color(c_fg)
        ax.title.set_color(c_fg)
        for spine in ax.spines.values(): spine.set_color(c_fg)
        
        # Apply to Text Labels (Main)
        # Note: Some are hardcoded colors (red/green), don't touch those.
        # Only touch defaults or gray ones.
        is_dark = (theme_name == 'Dark')
        txt_conn.set_color(c_fg if is_dark else 'gray')
        
        # [FIX] Force these to be visible
        gray_or_fg = 'white' if is_dark else 'gray' # Lighter for dark mode
        if txt_wear.get_text().endswith("--"): txt_wear.set_color(gray_or_fg)
        if txt_batt.get_text().endswith("--"): txt_batt.set_color(gray_or_fg)
        txt_audio.set_color(gray_or_fg)
        
        # txt_score has no color set, it defaults to black. Need to force it.
        txt_score.set_color(c_fg)
        
        # [FIX] Vol Axis
        ax_vol.set_facecolor(c_ax)
        ax_vol.tick_params(colors=c_fg) 
        ax_vol.title.set_color(c_fg)
        # Force redraw of ticks
        for t in ax_vol.yaxis.get_ticklabels(): t.set_color(c_fg)
        for spine in ax_vol.spines.values(): spine.set_color(c_fg)
        
        # Apply to Panels
        panels = [ax_trig_bg, ax_gui_bg, ax_proc_bg, ax_conf_bg, ax_audio_bg, ax_log_bg]
        for p in panels:
            p.set_facecolor(c_panel)
            # Find the title text (it's the only Artist in the ax_p usually, or stored separately)
            # In create_panel_ax, we did ax_p.text and didn't save ref.
            # We can iterate artists.
            for art in p.texts: art.set_color(c_fg)
        
        # [NEW] GSR View Theme
        if 'ax_gsr' in globals() or 'ax_gsr' in locals():
             # Check if ax_gsr is defined (it might be defined later if apply_theme call is early, 
             # but apply_theme is called at end of script)
             try:
                 ax_gsr.set_facecolor(c_bg) # Match window background or black? 
                 # Dial uses images, but if images fail, we need color.
                 # Actually ax_gsr has images on top. 
                 # Let's set texts.
                 txt_gsr_center.set_color(c_fg)
                 txt_gsr_inst.set_color(c_fg)
                 lbl_gsr_inst.set_color('gray') # Keep gray?
                 txt_sens.set_color(c_fg)
                 ax_gsr_back.set_facecolor(c_panel)
             except: pass
                
        # Radio Buttons / Checkboxes Backgrounds & Text
        radios = [
             (ax_thm, ui_refs['rad_theme']), 
             (ax_master_trig, ui_refs['chk_master']),
             (ax_rad, ui_refs['rad_source']),
             (ax_spk, ui_refs['chk_spike']),
             (ax_lchk, ui_refs['chk_light']),
             (ax_uchk, ui_refs['chk_use_color']) 
        ]
        
        # Add dynamic checkboxes (bands)
        for k in bands_history:
             if f'chk_{k}' in ui_refs:
                  # Find axis... it was created dynamically 'ax_chk'
                  # Hard to find parent axis for widget object directly without ref.
                  # But we can assume they are in 'settings_view_axes' or 'main_view_axes'?
                  # Actually they are panels in main view.
                  pass 

        for a, r in radios:
            a.set_facecolor(c_panel)
            try:
                for l in r.labels: l.set_color(c_fg)
            except: pass

        # Also update dynamic trigger checkboxes on Main View? No, they have specific band colors.
        # But their labels?
        # CheckButtons labels created with 'label_props' might be stuck.
        
        # [FIX] Bottom Status Bar (System Line)
        system_line.set_color(c_fg)
        
        # [FIX] Log Text
        log_text.set_color(c_fg)
        
        plt.draw()
        log_msg(f"Theme: {theme_name}")

    def on_theme_change(label):
        apply_theme(label)
    ui_refs['rad_theme'].on_clicked(on_theme_change)
    
    # 1. Triggers
    r_cn = [rect_trig[0] + 0.02, rect_trig[1] + 0.12, rect_trig[2] - 0.04, 0.03]
    ax_coin = reg_ax(r_cn, settings_view_axes)
    ui_refs['slide_win'] = Slider(ax_coin, 'Coin (s)', 0.1, 2.0, valinit=coincidence_window, color='gray')
    ui_refs['slide_win'].on_changed(lambda v: globals().update(coincidence_window=v))

    r_mas = [rect_trig[0] + 0.20, rect_trig[1] + 0.14, 0.08, 0.03]
    ax_master_trig = reg_ax(r_mas, settings_view_axes)
    ax_master_trig.set_facecolor('#e0e0e0'); ax_master_trig.axis('off')
    ui_refs['chk_master'] = CheckButtons(ax_master_trig, ['Active'], [triggers_enabled]) # [NEW] Use Loaded
    ui_refs['chk_master'].on_clicked(lambda l: globals().update(triggers_enabled=not triggers_enabled))

    r_bs = [rect_trig[0] + 0.02, rect_trig[1] + 0.08, rect_trig[2] - 0.04, 0.03]
    ax_base = reg_ax(r_bs, settings_view_axes)
    ui_refs['slide_base'] = Slider(ax_base, 'Base (s)', 0.1, 2.0, valinit=baseline_window_sec, color='lightblue')
    ui_refs['slide_base'].on_changed(lambda v: globals().update(baseline_window_sec=v))
    
    r_th = [rect_trig[0] + 0.02, rect_trig[1] + 0.04, rect_trig[2] - 0.04, 0.03]
    ax_thresh = reg_ax(r_th, settings_view_axes)
    ui_refs['slide_thresh'] = Slider(ax_thresh, 'Thresh %', 0, 50, valinit=global_percent, color='gold')
    ui_refs['slide_thresh'].on_changed(lambda v: globals().update(global_percent=v))

    # 2. Processing
    r_wn = [rect_proc[0] + 0.02, rect_proc[1] + 0.11, rect_proc[2] - 0.04, 0.03]
    ax_win = reg_ax(r_wn, settings_view_axes)
    ui_refs['slide_window'] = Slider(ax_win, 'Win (s)', 0.5, 4.0, valinit=current_window_sec, color='cyan')
    ui_refs['slide_window'].on_changed(lambda v: globals().update(current_window_sec=v)) 

    r_sm = [rect_proc[0] + 0.02, rect_proc[1] + 0.06, rect_proc[2] - 0.04, 0.03]
    ax_smooth = reg_ax(r_sm, settings_view_axes)
    ui_refs['slide_smooth'] = Slider(ax_smooth, 'Smooth', 0.1, 1.0, valinit=current_smoothing, color='magenta')
    ui_refs['slide_smooth'].on_changed(lambda v: globals().update(current_smoothing=v)) 
    
    r_spk = [rect_proc[0] + 0.02, rect_proc[1] + 0.02, rect_proc[2] - 0.04, 0.035]
    ax_spk = reg_ax(r_spk, settings_view_axes)
    ui_refs['chk_spike'] = CheckButtons(ax_spk, ['Spike Detect'], [spike_detection_enabled]) # [NEW] Use Loaded
    def toggle_spike_logic(label):
        global spike_detection_enabled
        spike_detection_enabled = ui_refs['chk_spike'].get_status()[0]
    ui_refs['chk_spike'].on_clicked(toggle_spike_logic)
    
    # 3. [NEW] Audio Input Control
    # Label (Current Mic)
    ax_mic_lbl = reg_ax([rect_audio[0]+0.02, rect_audio[1]+0.16, rect_audio[2]-0.04, 0.06], settings_view_axes)
    ax_mic_lbl.axis('off')
    ax_mic_lbl.axis('off')
    # [NEW] Start as "NO MIC" until resolved
    ui_refs['text_mic_name'] = ax_mic_lbl.text(0, 0.5, "NO MIC", va="center", ha="left", fontsize=9)
    
    # [NEW] Audio Meter Bar
    r_meter = [rect_audio[0]+0.02, rect_audio[1]+0.11, rect_audio[2]-0.09, 0.02] 
    ax_level = reg_ax(r_meter, settings_view_axes)
    ax_level.set_xlim(0, 1.0); ax_level.set_ylim(-0.5, 0.5) 
    ax_level.set_xticks([]); ax_level.set_yticks([])
    ax_level.set_facecolor("#333")
    ax_level.set_zorder(10) # FORCE ON TOP
    bar_level = ax_level.barh([0], [0], color='green', height=1.0)
    
    # [NEW] Level Text %
    ax_lvl_txt = reg_ax([rect_audio[0]+rect_audio[2]-0.06, rect_audio[1]+0.10, 0.05, 0.04], settings_view_axes)
    ax_lvl_txt.axis('off')
    ui_refs['text_level'] = ax_lvl_txt.text(0.5, 0.5, "0%", ha='center', va='center', fontsize=8, color='blue')

    # Button (Select) - Moved Down slightly
    r_msel = [rect_audio[0]+0.02, rect_audio[1]+0.06, 0.20, 0.035]
    ax_msel = reg_ax(r_msel, settings_view_axes)
    ui_refs['btn_select_mic'] = Button(ax_msel, 'Select Input...', color='lightyellow')
    ui_refs['btn_select_mic'].on_clicked(open_audio_select)

    # Slider (Gain) - Moved Down slightly
    r_gain = [rect_audio[0] + 0.02, rect_audio[1] + 0.01, rect_audio[2] - 0.04, 0.03]
    ax_gain = reg_ax(r_gain, settings_view_axes)
    ui_refs['slide_gain'] = Slider(ax_gain, 'Mic Gain', 1.0, 10.0, valinit=current_mic_gain, color='lime')
    ui_refs['slide_gain'].on_changed(lambda v: globals().update(current_mic_gain=v))

    # 4. DEVICE CONTROLS
    left_x = rect_conf[0] + 0.02
    
    r_lchk = [left_x, rect_conf[1] + 0.18, 0.18, 0.04]
    ax_lchk = reg_ax(r_lchk, settings_view_axes)
    ui_refs['chk_light'] = CheckButtons(ax_lchk, ['Light On'], [light_active]) # [NEW] Use Loaded State
    
    def toggle_light_cb(label):
        global light_active
        light_active = ui_refs['chk_light'].get_status()[0]
        if light_active:
            send_command(0, b'\x20\x00') 
            send_command(9, b'\x10\xff\xff\xff\xff\x0f') 
            log_msg("Light: ON")
        else:
            send_command(9, b'\x10\x00\x00\x00\x00\x00') 
            log_msg("Light: OFF")
    ui_refs['chk_light'].on_clicked(toggle_light_cb)
    
    r_uchk = [left_x, rect_conf[1] + 0.14, 0.18, 0.04]
    ax_uchk = reg_ax(r_uchk, settings_view_axes)
    ui_refs['chk_use_color'] = CheckButtons(ax_uchk, ['Use Colour'], [use_colour_active]) # [NEW] Use Loaded State
    
    def toggle_use_color(label):
        global use_colour_active
        use_colour_active = ui_refs['chk_use_color'].get_status()[0]
        log_msg(f"Biofeedback: {'ON' if use_colour_active else 'OFF'}")
    ui_refs['chk_use_color'].on_clicked(toggle_use_color)
    
    r_rad = [left_x + 0.18, rect_conf[1] + 0.13, 0.10, 0.10]
    ax_rad = reg_ax(r_rad, settings_view_axes)
    ax_rad.set_facecolor('#e0e0e0'); ax_rad.axis('off')
    # [NEW] Set active based on Loaded
    src_idx = 1 if biofeedback_source == 'Focus' else 0
    ui_refs['rad_source'] = RadioButtons(ax_rad, ['Calm', 'Focus'], active=src_idx)
    
    def change_source(label):
        global biofeedback_source
        biofeedback_source = label
        log_msg(f"Source: {label}")
    ui_refs['rad_source'].on_clicked(change_source)
    
    # Vibration (Lower)
    r_vib = [rect_conf[0] + 0.02, rect_conf[1] + 0.045, rect_conf[2] - 0.04, 0.03]
    ax_vib = reg_ax(r_vib, settings_view_axes)
    ui_refs['slide_str'] = Slider(ax_vib, 'Vib (0-100)', 0, 100, valinit=0, color='purple')
    
    def on_vib_change(val):
        strength = int(val)
        b = bytearray([0x01, strength]) 
        send_command(0x13, b)
    ui_refs['slide_str'].on_changed(on_vib_change)

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

    # [MOVED] teleport calls moved to end of setup
    # teleport_off(settings_view_axes) 
    # teleport_on(main_view_axes)

    def req_main(e): 
        global desired_view; desired_view = 'main'
        sync_audio_stream('main')
    def req_settings(e): 
        global desired_view; desired_view = 'settings'
        sync_audio_stream('settings')
    
    ui_refs['btn_to_settings'].on_clicked(req_settings)
    ui_refs['btn_back'].on_clicked(req_main)
    
    # --- GSR VIEW SETUP ---
    # We use a full-screen axis mapped to 1024x768 to match original logic
    # Note: 14x9 inches -> Aspect 1.55. 1024/768 -> 1.33. We will just stretch or center.
    # Let's map 0..1024 and 0..768 directly.
    ax_gsr = reg_ax([0.0, 0.0, 1.0, 1.0], gsr_view_axes)
    ax_gsr.set_xlim(0, 1024); ax_gsr.set_ylim(768, 0) # Inverted Y
    ax_gsr.axis('off')
    
    # Load Images
    try:
        gsr_bg_img = plt.imread("Background.jpg")
        ax_gsr.imshow(gsr_bg_img, extent=[0, 1024, 768, 0], aspect='auto')
    except: pass
    
    try:
        gsr_dial_img = plt.imread("dial_background.png")
        # Scale pos: 512, 195. Extent needs L, R, B, T (or T, B for inverted?)
        # Matplotlib extent: [left, right, bottom, top]. Y is inverted (768 at bottom).
        # So [left, right, 768, 0] ? No, standard order with inverted axis.
        # Center is 512, 195. Let's assume image size is relevant.
        # Img size unknown? We'll guess or use scatter for pivot.
        # Actually gsr_dial_v14 used canvas.create_image(512, 195).
        # We can use an AnnotationBbox or inset axes, or just imshow with bounds.
        # For simplicity, let's just draw the needle and graph. The dial background might be tricky to overlay perfectly without size.
        # We will try to load it.
        h_d, w_d = gsr_dial_img.shape[:2]
        # Center 512, 195. 
        extent_d = [512 - w_d/2, 512 + w_d/2, 195 + h_d/2, 195 - h_d/2] # Y inverted
        ax_gsr.imshow(gsr_dial_img, extent=extent_d)
    except: pass

    # NEEDLE
    needle_line, = ax_gsr.plot([], [], color="#882222", lw=4, solid_capstyle='round')
    
    # GRAPH BOX
    # 512, 500. W=700, H=150.
    g_x, g_y, g_w, g_h = (1024-700)//2, 500, 700, 150
    rect_graph = plt.Rectangle((g_x, g_y - g_h/2), g_w, g_h, facecolor='#dddddd', edgecolor='#999', lw=2)
    ax_gsr.add_patch(rect_graph)
    gsr_graph_line, = ax_gsr.plot([], [], color='#00aa00', lw=2)
    
    # READOUTS
    txt_gsr_center = ax_gsr.text(180, 768-100, "2.00", fontsize=30, fontweight='bold', color='black', ha='center')
    ax_gsr.text(180, 768-130, "TA POSITION", fontsize=10, fontweight='bold', color='#555', ha='center')
    
    txt_gsr_inst = ax_gsr.text(1024-180, 768-100, "0.000", fontsize=30, fontweight='bold', color='black', ha='center')
    lbl_gsr_inst = ax_gsr.text(1024-180, 768-130, "INSTANT TA", fontsize=10, fontweight='bold', color='#555', ha='center')
    
    # CONTROLS - CUSTOM UI (Patches)
    # 1. Counter Box (20, 20) -> (250, 100)
    gsr_rect_counter = plt.Rectangle((20, 20), 230, 80, facecolor='#ffcccc', edgecolor='#333', lw=2)
    ax_gsr.add_patch(gsr_rect_counter)
    gsr_txt_cnt_status = ax_gsr.text(135, 40, "COUNTING: PAUSED", fontsize=10, fontweight='bold', color='#550000', ha='center', va='center')
    gsr_txt_cnt_val = ax_gsr.text(135, 70, "0.00", fontsize=28, fontweight='bold', color='black', ha='center', va='center')
    
    # 2. Reset Button (20, 105) -> (250, 135)
    gsr_rect_reset = plt.Rectangle((20, 105), 230, 30, facecolor='#dddddd', edgecolor='#555', lw=1)
    ax_gsr.add_patch(gsr_rect_reset)
    ax_gsr.text(135, 120, "RESET", fontsize=9, fontweight='bold', color='#333', ha='center', va='center')
    
    # 3. Booster (Right Top)
    # Box: (1024-300, 20) -> (1024-20, 70). Width 280.
    bx_start = 1024 - 300
    gsr_rect_boost_bg = plt.Rectangle((bx_start, 20), 280, 50, facecolor='#eeeeee', edgecolor='#333', lw=2)
    ax_gsr.add_patch(gsr_rect_boost_bg) # Background box? Or individual segments? 
    # Original drawn segments. Let's do segments.
    # Actually original had a container box too? "tags='overlay'". 
    # "create_rectangle(bx_start, 20, bx_end, 70...)" -> Background.
    # Then segments. 
    # Let's just do increments.
    ax_gsr.text(bx_start + 140, 12, "AUTO-BOOSTER", fontsize=8, fontweight='bold', color='#333', ha='center')
    
    gsr_boost_rects = []
    gsr_boost_texts = []
    seg_w = 280 / 4
    b_labels = ["OFF", "LO", "MED", "HI"]
    for i in range(4):
        x = bx_start + (i * seg_w)
        r = plt.Rectangle((x, 20), seg_w, 50, facecolor='#dddddd', edgecolor='#999', lw=1)
        ax_gsr.add_patch(r)
        txt_b = ax_gsr.text(x + seg_w/2, 45, b_labels[i], fontsize=10, fontweight='normal', color='#777', ha='center', va='center')
        gsr_boost_rects.append(r)
        gsr_boost_texts.append(txt_b)
        
    # 4. Sensitivity Bar (Bottom Center)
    # Width depends on zoom. Max width 200px?
    # Center: 512, 768-35. (Y inverted: 768 at bottom. So 768-40 to 768-30)
    # rect(WIN_W / 2 - bar_w, WIN_H - 40, ...)
    gsr_rect_sens = plt.Rectangle((512, 768-40), 0, 10, facecolor='blue')
    ax_gsr.add_patch(gsr_rect_sens)
    
    # [RESTORED] Sensitivity Text
    txt_sens = ax_gsr.text(512, 768-20, "SENSITIVITY: 0.20", fontsize=10, fontweight='bold', ha='center', color='black') # Default color


    # CLICK HANDLER
    def on_gsr_click(event):
        global gsr_counting_active, gsr_ta_total, gsr_center_ta, gsr_booster_level
        if current_view != 'gsr' or event.inaxes != ax_gsr: return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        
        # 1. Counter (20, 20, 230, 80)
        if 20 <= x <= 250 and 20 <= y <= 100:
            gsr_counting_active = not gsr_counting_active
            
        # 2. Reset (20, 105, 230, 30) (Only if not counting)
        elif 20 <= x <= 250 and 105 <= y <= 135:
            if not gsr_counting_active:
                gsr_ta_total = 0.0
                gsr_center_ta = gsr_current_ta # Also center on reset? Original logic: recenter if space or auto.
                # Actually Reset button just resets Total. 
                # Spacebar centers. 
                # Let's make Reset assume recenter too? Original: "reset_counter" just zeroes ta_total.
                gsr_ta_total = 0.0
                
        # 3. Booster (bx_start, 20, 280, 50)
        elif bx_start <= x <= (bx_start + 280) and 20 <= y <= 70:
            # Which segment?
            rel_x = x - bx_start
            idx = int(rel_x // (280/4))
            if 0 <= idx <= 3: gsr_booster_level = idx
            
        # 4. Center on Click (Anywhere else? Or specific?)
        # Original: Space bar centers.
        pass

    fig.canvas.mpl_connect('button_press_event', on_gsr_click)


    # Navigation (To GSR)
    # Add GSR button to Main View
    r_to_gsr = [0.05, 0.06, 0.05, 0.04] # Replaces Re-Calib? No, shift it.
    # We need space. Let's put it next to Settings.
    r_g = [0.65, 0.06, 0.10, 0.04]
    ax_to_gsr = reg_ax(r_g, main_view_axes)
    btn_to_gsr = Button(ax_to_gsr, "GSR View", color='lightblue')
    
    def req_gsr(e):
        global desired_view; desired_view = 'gsr'
        sync_audio_stream('gsr') # Audio should probably be OFF or handled
    btn_to_gsr.on_clicked(req_gsr)
    
    # Back button for GSR
    r_gbk = [0.02, 0.02, 0.08, 0.04]
    ax_gsr_back = reg_ax(r_gbk, gsr_view_axes)
    btn_gsr_back = Button(ax_gsr_back, "< Back", color='lightgray')
    btn_gsr_back.on_clicked(req_main)
    
    # teleport_off(gsr_view_axes) # [MOVED] Handled globally below

    # [NEW] Final Initial View State
    teleport_off(settings_view_axes)
    teleport_off(gsr_view_axes)
    teleport_on(main_view_axes)

    dev_static = {}
    last_color_update_time = 0
    
    # --- AUTO-DETECT MIC ON STARTUP REMOVED (Handled by sync_audio_stream + load_config)

    def update(frame):
        global current_view, desired_view, current_state, event_detected, ignore_ui_callbacks, last_color_update_time
        
        if is_connected:
            if time.time() - last_packet_time > 1.5: current_state = "LINK : DISCONNECTED"
            else: current_state = "LINK : STREAMING"
        else: current_state = "LINK : DISCONNECTED"
        
        if is_recording: rec_text.set_alpha(1.0 if frame % 10 < 5 else 0.3)
        else: rec_text.set_alpha(0.0)
        
        if spike_detection_enabled and time.time() - last_spike_time < 1.0:
            spike_text.set_text(f"⚡ SPIKE\nMag: {current_spike_mag}")
            spike_text.set_visible(True)
        else: spike_text.set_visible(False)

        if current_view != desired_view:
            teleport_off(main_view_axes); teleport_off(settings_view_axes); teleport_off(gsr_view_axes)
            if desired_view == 'settings': teleport_on(settings_view_axes)
            elif desired_view == 'main': teleport_on(main_view_axes)
            elif desired_view == 'gsr': teleport_on(gsr_view_axes)
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
        
        if is_connected and light_active and use_colour_active:
            now = time.time()
            if now - last_color_update_time > 0.25:
                score = current_calm_score if biofeedback_source == 'Calm' else current_focus_score
                pkt = get_color_pkt(score, biofeedback_source)
                send_command(9, pkt)
                last_color_update_time = now

        log_text.set_text("\n".join(list(log_messages))) 
        s_str = f"Name: {dev_static.get('name','?')} | SN: {dev_static.get('serial','?')} | HW: {dev_static.get('hw','?')} | FW: {dev_static.get('fw','?')} | Manuf: {dev_static.get('manuf','?')} | Batt: {device_battery_level} | Mic: {current_mic_name}"
        system_line.set_text(s_str)
        txt_conn.set_text(current_state)
        
        if current_state == "LINK : STREAMING":
            txt_wear.set_text("HEADSET : ON HEAD" if headset_on_head else "HEADSET : OFF HEAD")
            txt_wear.set_color("lime" if headset_on_head else "red") # Explicit colors for active state
            txt_batt.set_text(f"BATT: {device_battery_level}")
            
            # [FIX] Ensure these revert to theme color if not active? 
            # Actually BATT should probably be theme color (white/black) unless critical?
            # Let's keep BATT as theme color.
            is_dark = (current_theme == 'Dark')
            c_std = 'white' if is_dark else 'gray'
            txt_batt.set_color(c_std)
            
            txt_score.set_text(f"Focus: {current_focus_score} | Calm: {current_calm_score}")
        else:
             # Disconnected state
             is_dark = (current_theme == 'Dark')
             c_std = 'white' if is_dark else 'gray'
             
             txt_wear.set_text("HEADSET : --"); txt_wear.set_color(c_std)
             txt_batt.set_text("BATT: --"); txt_score.set_text("Score: --")
             txt_batt.set_color(c_std)

        should_update_graph = (current_state == "LINK : STREAMING") and headset_on_head

        should_update_graph = (current_state == "LINK : STREAMING") and headset_on_head
        
        if should_update_graph:
            vis_vals = []
            bands_checked = 0
            recent_triggers = 0
            now = time.time()
            
            for k in bands_history:
                dlist = list(bands_history[k])
                
                if triggers[k]['active']:
                    vis_vals.extend(dlist)
                    if current_view == 'main':
                        lines[k].set_data(range(len(dlist)), dlist)
                        lines[k].set_visible(True) 
                        if triggers_enabled:
                            bands_checked += 1
                            if len(dlist) > 20: 
                                recent = dlist[-20:]; mean = np.mean(recent)
                                mean_lines[k].set_ydata([mean]); mean_lines[k].set_visible(True)
                                
                                percent = global_percent / 100.0
                                mode = triggers[k]['mode']
                                thresh = mean * (1.0 + percent) if mode == 1 else max(0, mean * (1.0 - percent))
                                thresh_lines[k].set_ydata([thresh]); thresh_lines[k].set_visible(True)
                                
                                if len(dlist) > 0:
                                    val = dlist[-1]
                                    if (mode == 1 and val > thresh) or (mode == -1 and val < thresh):
                                         triggers[k]['last_seen'] = now
                            else:
                                 mean_lines[k].set_visible(False)
                                 thresh_lines[k].set_visible(False)
                            if now - triggers[k]['last_seen'] < coincidence_window: recent_triggers += 1
                        else:
                            mean_lines[k].set_visible(False)
                            thresh_lines[k].set_visible(False)
                else: 
                     if current_view == 'main':
                         lines[k].set_visible(False)
                         mean_lines[k].set_visible(False)
                         thresh_lines[k].set_visible(False)

            if bands_checked > 0 and recent_triggers == bands_checked:
                event_detected = True; event_text.set_visible(True)
            else:
                event_detected = False; event_text.set_visible(False)
                
            if vis_vals:
                mx = max(vis_vals)
                ax.set_ylim(0, max(0.1, mx * 1.1)) 
            else:
                ax.set_ylim(0, 100)

            if len(eeg_buffer) >= volatility_window:
                recent_raw = list(itertools.islice(eeg_buffer, len(eeg_buffer) - volatility_window, len(eeg_buffer)))
                current_vol = np.std(recent_raw)
                if current_vol > max_vol_scale: vol_bar[0].set_height(max_vol_scale)
                else: vol_bar[0].set_height(current_vol)

                if current_vol > 500: vol_bar[0].set_color('magenta')
                elif current_vol > 100: vol_bar[0].set_color('red')
                else: vol_bar[0].set_color('cyan')

        # Update Audio Meter (INDEPENDENT of EEG State)
        if audio_stream and audio_stream.active:
             # VISUAL GAIN REMOVED (Raw 1:1)
             raw_lvl = audio_state['peak']
             lvl = min(1.0, raw_lvl) 
             
             bar_level[0].set_width(lvl)
             ui_refs['text_level'].set_text(f"{int(lvl*100)}%") 
             
             if lvl > 0.95: bar_level[0].set_color('red') # Near Peak
             elif lvl > 0.6: bar_level[0].set_color('yellow')
             else: bar_level[0].set_color('lime')
        else:
             bar_level[0].set_width(0)
             ui_refs['text_level'].set_text("OFF")

        artists = []
        if current_view == 'main':
             # Main View Artists
             artists.extend([rec_text, spike_text, event_text, txt_conn, txt_wear, txt_batt, txt_score, vol_bar[0]])
             for k in bands_history:
                 if triggers[k]['active']:
                     artists.append(lines[k])
                     artists.append(mean_lines[k])
                     artists.append(thresh_lines[k])

        elif current_view == 'settings':
             # Settings View Artists (Explicitly add Meter)
             artists.extend([bar_level[0], ui_refs['text_level'], log_text, system_line])
        
        elif current_view == 'gsr':
             # --- GSR LOGIC ---
             global gsr_last_ta_frame, gsr_is_artifact, gsr_center_ta, gsr_ta_total, gsr_counting_active, gsr_booster_level
             
             ta = gsr_current_ta
             
             # Artifact
             delta = abs(ta - gsr_last_ta_frame)
             if delta > 0.05: gsr_is_artifact = True
             else: gsr_is_artifact = False
             gsr_last_ta_frame = ta
             
             # Zoom
             eff_zoom = gsr_base_sensitivity
             if gsr_booster_level == 1: eff_zoom *= math.pow((2/max(1, gsr_center_ta)), 0.6)
             elif gsr_booster_level == 2: eff_zoom *= math.pow((2/max(1, gsr_center_ta)), 1.0)
             elif gsr_booster_level == 3: eff_zoom *= math.pow((2/max(1, gsr_center_ta)), 1.4)
             
             # Angle
             half_win = eff_zoom / 2.0
             diff = ta - gsr_center_ta
             ratio = diff / half_win
             # Dial settings: Center 90 (Up). Scale -40 to +40 degrees sweep.
             # In Matplotlib (radians), 0 is Right, 90 is Up.
             angle_deg = 90 - ((ratio * 40) + 11) # +11 offset from original
             
             # Auto-Center (Wrap)
             if abs((ratio * 40)) > 40: 
                  old_center = gsr_center_ta
                  gsr_center_ta = ta # Update Center
                  diff = ta - gsr_center_ta
                  ratio = diff / half_win
                  
                  # Add to Total if counting
                  if gsr_counting_active:
                       if ta < old_center: gsr_ta_total += (old_center - ta)
             
             # Calculate final angle again
             angle_deg = 90 - ((ratio * 40) + 11)

             
             angle_rad = math.radians(angle_deg)
             
             # Update Needle
             # Pivot: 512, 575
             px, py = 512, 575
             tip_x = px + 470 * math.cos(angle_rad)
             tip_y = py - 470 * math.sin(angle_rad) # Y inverted
             needle_line.set_data([px, tip_x], [py, tip_y])
             
             # Graph
             # Height 150. Middle 500.
             # ratio is -1 to 1 basically.
             # Original: plot_y = graph_ratio * (GRAPH_H / 2) * GRAPH_SCALE
             # Inverted Y: 500 - val.
             val = ((ratio * 40 + 11) / 40) * (150/2) * 0.9
             gsr_graph_data.append(val)
             
             # Draw Graph
             g_x_start = (1024-700)//2
             pts_x = np.linspace(g_x_start, g_x_start+700, len(gsr_graph_data))
             pts_y = [500 - v for v in gsr_graph_data]
             gsr_graph_line.set_data(pts_x, pts_y)
             if gsr_is_artifact: gsr_graph_line.set_color('orange')
             else: gsr_graph_line.set_color('#00aa00')
             
             # Strings
             txt_gsr_center.set_text(f"{gsr_center_ta:.2f}")
             txt_gsr_inst.set_text(f"{ta:.3f}")
             if gsr_is_artifact:
                 lbl_gsr_inst.set_text("BODY MOTION"); lbl_gsr_inst.set_color('orange')
                 txt_gsr_inst.set_color('orange')
             else:
                 lbl_gsr_inst.set_text("INSTANT TA"); lbl_gsr_inst.set_color('#555')
                 txt_gsr_inst.set_color('black')
                 
             txt_sens.set_text(f"SENSITIVITY: {eff_zoom:.2f}")
             if gsr_booster_level > 0: txt_sens.set_text(f"SENSITIVITY: {eff_zoom:.2f} (AUTO)")
             
             # UI UPDATES (Patches)
             # Counter
             if gsr_counting_active:
                 gsr_rect_counter.set_facecolor('#ccffcc')
                 gsr_txt_cnt_status.set_text("COUNTING: ON"); gsr_txt_cnt_status.set_color('#005500')
             else:
                 gsr_rect_counter.set_facecolor('#ffcccc')
                 gsr_txt_cnt_status.set_text("COUNTING: PAUSED"); gsr_txt_cnt_status.set_color('#550000')
             gsr_txt_cnt_val.set_text(f"{gsr_ta_total:.2f}")
             
             # Booster
             b_cols = ["#999999", "#eeee00", "#00aa00", "#00aaaa"]
             for i, (r, txt_obj) in enumerate(zip(gsr_boost_rects, gsr_boost_texts)):
                 is_act = (gsr_booster_level == i)
                 r.set_facecolor(b_cols[i] if is_act else '#dddddd')
                 txt_obj.set_color('black' if is_act else '#777777')
                 txt_obj.set_fontweight('bold' if is_act else 'normal')
                 
             # Sens Bar
             # Width = (1.0 / eff_zoom) * 25. Centered at 512.
             bar_w = int((1.0 / eff_zoom) * 25)
             gsr_rect_sens.set_x(512 - bar_w)
             gsr_rect_sens.set_width(bar_w * 2)

             artists.extend([needle_line, gsr_graph_line, txt_gsr_center, txt_gsr_inst, txt_sens, lbl_gsr_inst,
                             gsr_rect_counter, gsr_txt_cnt_status, gsr_txt_cnt_val, gsr_rect_reset, gsr_rect_sens] + gsr_boost_rects + gsr_boost_texts)


        return artists

    def on_close(event):
        global app_running
        app_running = False
        save_config() # [NEW] Save on Exit
        print("Window Closed. Cleaning up...")

    fig.canvas.mpl_connect('close_event', on_close)

    try:
        # DISABLE BLIT - Fix Artifacts
        ani = animation.FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False) 
        
        # [moved] Force Device Resolution AFTER GUI is built
        try:
             sync_audio_stream('main') 
        except: pass
        
        # [NEW] Apply Initial Theme
        if current_theme: apply_theme(current_theme)

        plt.show()
    except: pass
    finally:
        app_running = False
        print("Stopping Bluetooth...")
        if csv_file: csv_file.close()
        if csv_file_detailed: csv_file_detailed.close()
        t.join(timeout=2.0)
        print("Shutdown Complete.")
