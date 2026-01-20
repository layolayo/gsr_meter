
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

# --- CONFIGURATION ---
FS = 250 # Hertz of headset datastream
MAX_WINDOW_SEC = 10
BUFFER_SIZE = int(FS * MAX_WINDOW_SEC)
HISTORY_LEN = 2500 
WARMUP_SAMPLES = FS * 3
SCORE_SMOOTHING_WINDOW = int(FS * 2.0) 
TREND_WINDOW_SEC = 10.0
TREND_WINDOW_SAMPLES = int(FS * TREND_WINDOW_SEC) 

# File Naming
FILENAME_MAIN = "brainwave_session"
FILENAME_DETAILED = "brainwave_detailed"
CONFIG_FILE = "config_v28.json" 

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
current_state = "LINK : DISCONNECTED" 
current_theme = "Light" 
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
calm_history = collections.deque(maxlen=SCORE_SMOOTHING_WINDOW)

# Logic Settings
event_detected = False

# Control Defaults
FFT_WINDOW_SEC = 1.0  
baseline_window_sec = 2.0
coincidence_window = 0.5
global_percent = 20
triggers_enabled = False

# Volatility Settings
volatility_window = 25  
max_vol_scale = 2000.0  

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
    global current_client, timestamp_queue
    global last_on_signal_time, total_samples_recorded

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
                req_samples = int(FS * FFT_WINDOW_SEC)
                bands = [0]*5; calm_inst=0
                if headset_on_head and is_warmed_up and len(eeg_buffer) >= req_samples:
                     bands = calculate_relative_bands(list(itertools.islice(eeg_buffer, len(eeg_buffer) - req_samples, len(eeg_buffer))))
                     try:
                         # Calculate Calm Score
                         raw_c = 30 * math.log10(bands[1] / (bands[2] + max(1e-6, bands[3]))) + 50
                         calm_inst = int(np.clip(raw_c, 0, 100))
                     except: pass
                     calm_history.append(calm_inst)
                     
                     if i == len(filt_chunk) - 1:
                         global current_calm_score
                         if len(calm_history) > 0: current_calm_score = int(np.mean(calm_history))
                         else: current_calm_score = calm_inst
                    
                     for k, b_val in zip(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], bands):
                         bands_history[k].append(b_val)
                
                if is_recording and csv_writer_detailed and recording_start_time:
                     sample_idx = total_samples_recorded + i
                     rel_ts_val = sample_idx / FS
                     abs_ts = (recording_start_time + timedelta(seconds=rel_ts_val)).strftime('%H:%M:%S.%f')
                     rel_ts = f"{rel_ts_val:.4f}"
                     detailed_rows.append([
                         abs_ts, rel_ts, new_raw[i], int(headset_on_head),
                         bands[0], bands[1], bands[2], bands[3], bands[4],
                         current_calm_score # Removed Focus Score
                     ])

            if detailed_rows:
                try: 
                    csv_writer_detailed.writerows(detailed_rows)
                    total_samples_recorded += len(detailed_rows)
                except: pass

            if is_recording and csv_writer:
                ts = datetime.now().strftime('%H:%M:%S.%f')
                # Removed Focus score from Main CSV as well
                row = [ts, current_state, int(headset_on_head), 1 if event_detected else 0, 0, current_calm_score] + bands + list(new_raw[:50])
                csv_writer.writerow(row)

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
            'bt_address': DEVICE_ADDRESS,
            'bt_name': ADVERTISED_NAME, 
            'gui_theme': current_theme
        }
            
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
        
        # Restore Theme
        global current_theme
        current_theme = cfg.get('gui_theme', 'Light')
        
        # Restore Bluetooth
        saved_addr = cfg.get('bt_address', None)
        saved_name = cfg.get('bt_name', "Unknown")
        if saved_addr:
             DEVICE_ADDRESS = saved_addr
             ADVERTISED_NAME = saved_name
             print(f"[Config] Restored Target Device: {ADVERTISED_NAME} ({DEVICE_ADDRESS})")
        
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
            log_msg(f"[SCAN] Scanning for {ADVERTISED_NAME if ADVERTISED_NAME else 'Devices'}...")
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
            if not is_connected: await asyncio.sleep(2.0)
        except: await asyncio.sleep(2.0)

def run_ble():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bluetooth_task())

if __name__ == "__main__":
    t = threading.Thread(target=run_ble, daemon=True)
    t.start()
    
    load_config() 
    pass
    
    # --- GUI ---
    fig = plt.figure(figsize=(14, 9))
    try: fig.canvas.manager.set_window_title("EEG Rolling Trend Viewer")
    except: pass 
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    current_view = 'main'
    desired_view = 'main'
    ax_positions = {}
    main_view_axes = []
    settings_view_axes = []
    
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
    # Replaces bars and states
    ax_graph = reg_ax([0.05, 0.15, 0.90, 0.75], main_view_axes)
    ax_graph.set_xlim(0, HISTORY_LEN)
    ax_graph.set_ylim(-5, 105)
    ax_graph.grid(True, alpha=0.3)
    ax_graph.set_title("Real-Time EEG Trends (Last ~10s)", fontsize=14, fontweight='bold')
    ax_graph.set_ylabel("Relative Power (%)")
    # ax_graph.set_xlabel("Time (samples)") # Hide for cleaner look
    ax_graph.set_xticks([])

    # Create lines
    lines = {}
    colors = {'Delta': 'blue', 'Theta': 'green', 'Alpha': 'orange', 'Beta': 'red', 'Gamma': 'purple'}
    
    for k, col in colors.items():
        line, = ax_graph.plot([], [], lw=1.5, color=col, label=k)
        lines[k] = line
    
    ax_graph.legend(loc='upper right', ncol=5)

    # === SCORES DISPLAY (Below Graph) ===
    # Kept Calm score, removed Focus
    ax_scores = reg_ax([0.40, 0.05, 0.20, 0.08], main_view_axes)
    ax_scores.set_xlim(0, 1)
    ax_scores.set_ylim(0, 1)
    ax_scores.set_xticks([])
    ax_scores.set_yticks([])
    ax_scores.set_facecolor('#e8e8e8')
    txt_calm_score = ax_scores.text(0.5, 0.5, "Calm: --", ha='center', va='center', fontsize=20, fontweight='bold')
    
    # === STATUS BAR (Top) ===
    ax_status = reg_ax([0.05, 0.94, 0.90, 0.04], main_view_axes)
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    ax_status.set_facecolor('#333')
    
    txt_conn = ax_status.text(0.02, 0.5, "LINK: DISCONNECTED", color='gray', fontsize=10, fontweight='bold', va='center')
    txt_wear = ax_status.text(0.25, 0.5, "HEADSET: --", color='gray', fontsize=10, fontweight='bold', va='center')
    txt_batt = ax_status.text(0.45, 0.5, "BATT: --", color='white', fontsize=10, fontweight='bold', va='center')
    txt_audio = ax_status.text(0.60, 0.5, "AUDIO: --", color='gray', fontsize=10, fontweight='bold', va='center')
    rec_text = ax_status.text(0.85, 0.5, "● REC", color='red', fontsize=10, fontweight='bold', va='center', visible=False)
    
    ui_refs = {}

    r_rl = [0.05, 0.06, 0.12, 0.04]
    ax_relock = reg_ax(r_rl, main_view_axes)
    ui_refs['btn_relock'] = Button(ax_relock, 'Clear Graph', color='lightblue')
    def clear_graph(e):
        for k in lines: bands_history[k].clear()
    ui_refs['btn_relock'].on_clicked(clear_graph)

    r_rc = [0.20, 0.06, 0.12, 0.04]
    ax_rec = reg_ax(r_rc, main_view_axes)
    ui_refs['btn_rec'] = Button(ax_rec, 'Record', color='lightgreen')

    
    import tkinter as tk
    from tkinter import simpledialog, ttk

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
            
            rates_to_try = [44100, 48000, None] 
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
                    log_msg(f"Audio Stream: ON ({sr if sr else 'Auto'} Hz)")
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

    def toggle_rec(e):
        global is_recording, csv_file, csv_writer, csv_file_detailed, csv_writer_detailed, recording_start_time
        global audio_stream, audio_buffer, audio_filename, total_samples_recorded, selected_device_idx
        
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
             
             notes = note_data["text"]
             
             DATA_DIR = "Session_Data"
             os.makedirs(DATA_DIR, exist_ok=True) 
             
             ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
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
                 # [MODIFIED] Header: Removed Focus_Score
                 header = ["Timestamp", "State", "Headset_On", "Event_Detected", "Spike_Detected",
                           "Calm_Score", "Delta", "Theta",
                           "Alpha", "Beta", "Gamma"] + [f"Raw_{i}" for i in range(50)]
                 csv_writer.writerow(header)

                 csv_file_detailed = open(fname_detail, 'w', newline='')
                 csv_writer_detailed = csv.writer(csv_file_detailed)
                 # [MODIFIED] Header: Removed Focus_Score
                 header_detail = ["Abs_Time", "Rel_Time", "Raw_24bit", "Headset", "Delta", "Theta", "Alpha",
                                  "Beta", "Gamma", "Calm_Score"]
                 csv_writer_detailed.writerow(header_detail)

                 audio_buffer = []
                 is_recording = True 
                 sync_audio_stream(current_view)
                 
                 recording_start_time = datetime.now()
                 total_samples_recorded = 0
                 
                 ui_refs['btn_rec'].label.set_text("Stop")
                 ui_refs['btn_rec'].color = 'salmon'
                 rec_text.set_visible(True)
                 log_msg(f"Started: {ts_str}")
             except Exception as ex: log_msg(f"Error: {ex}")
        else:
             is_recording = False
             if csv_file: csv_file.close()
             if csv_file_detailed: csv_file_detailed.close()
             save_audio()
             sync_audio_stream(current_view) 
             
             ui_refs['btn_rec'].label.set_text("Record")
             ui_refs['btn_rec'].color = 'lightgreen'
             rec_text.set_visible(False)
             log_msg("Recording Saved")

    ui_refs['btn_rec'].on_clicked(toggle_rec)

    r_ts = [0.80, 0.06, 0.15, 0.04]
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
    ui_refs['rad_theme'] = RadioButtons(ax_thm, ['Light', 'Dark'], active=(1 if is_dark else 0))
    
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
            txt_calm_score.set_color(c_fg)
        except: pass
        
        # Apply to Status Bar
        try:
            ax_status.set_facecolor('#333' if theme_name == 'Dark' else '#ddd')
            txt_conn.set_color(c_fg if theme_name == 'Dark' else 'gray')
            txt_wear.set_color(c_fg if theme_name == 'Dark' else 'gray')
            txt_batt.set_color(c_fg)
            txt_audio.set_color(c_fg if theme_name == 'Dark' else 'gray')
        except: pass
        
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
    
    def update(frame):
        global current_view, desired_view, current_state, event_detected, ignore_ui_callbacks
        
        if is_connected:
            if time.time() - last_packet_time > 1.5: current_state = "LINK : DISCONNECTED"
            else: current_state = "LINK : STREAMING"
        else: current_state = "LINK : DISCONNECTED"
        
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
        txt_conn.set_text(current_state)
        
        if current_state == "LINK : STREAMING":
            txt_wear.set_text("HEADSET : ON HEAD" if headset_on_head else "HEADSET : OFF HEAD")
            txt_wear.set_color("lime" if headset_on_head else "red") 
            txt_batt.set_text(f"BATT: {device_battery_level}")
            
            is_dark = (current_theme == 'Dark')
            c_std = 'white' if is_dark else 'gray'
            txt_batt.set_color(c_std)
            
            txt_calm_score.set_text(f"Calm: {current_calm_score}")
        else:
             is_dark = (current_theme == 'Dark')
             c_std = 'white' if is_dark else 'gray'
             
             txt_wear.set_text("HEADSET : --"); txt_wear.set_color(c_std)
             txt_batt.set_text("BATT: --"); txt_calm_score.set_text("Calm: --")
             txt_batt.set_color(c_std)

        should_update_graph = (current_state == "LINK : STREAMING") and headset_on_head
        
        if should_update_graph and current_view == 'main':
             # UPDATE GRAPH
             for k, line in lines.items():
                 data = list(bands_history[k])
                 if len(data) > 0:
                     line.set_data(np.arange(len(data)), data)
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

        artists = []
        if current_view == 'main':
             artists.extend([rec_text, txt_conn, txt_wear, txt_batt])
             artists.extend(list(lines.values()))
             artists.extend([txt_calm_score])

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
        ani = animation.FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False) 
        
        try:
             sync_audio_stream('main') 
        except Exception as e:
             print(f"Audio sync error: {e}")
        
        if current_theme: apply_theme(current_theme)

        plt.show()
    except Exception as e:
        import traceback
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        app_running = False
        print("Stopping Bluetooth...")
        if csv_file: csv_file.close()
        if csv_file_detailed: csv_file_detailed.close()
        t.join(timeout=2.0)
        print("Shutdown Complete.")
