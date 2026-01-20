import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import hid
import threading
import time
import math
import sys
import os
import collections
import csv
import datetime
import asyncio
import queue
import numpy as np
from bleak import BleakClient, BleakScanner
from scipy.signal import butter, lfilter, lfilter_zi
import itertools
import json
import random # [NEW]
import sounddevice as sd # [NEW]
import scipy.io.wavfile as wav # [NEW]

# --- INTEGRATED GLOBALS ---
latest_gsr_ta = 0.0
latest_gsr_motion = False
latest_gsr_sens = 0.20
latest_gsr_center = 2.0

# --- EEG CONFIGURATION ---
FS = 250 # Hertz
BUFFER_SIZE = int(FS * 10)
HISTORY_LEN = 2500 
WARMUP_SAMPLES = FS * 3
SCORE_SMOOTHING_WINDOW = int(FS * 2.0) 

# --- UUIDS ---
UUID_BATTERY = "00002a19-0000-1000-8000-00805f9b34fb"
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

# --- EEG STATE ---
eeg_buffer = collections.deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
bands_history = {k: collections.deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN) for k in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']}
log_messages = collections.deque(["System Ready"], maxlen=10)
is_connected = False
current_state = "LINK : DISCONNECTED"
headset_on_head = False
app_running = True
last_packet_time = 0
samples_since_contact = 0
device_battery_level = "--"
current_client = None
timestamp_queue = None
REAL_SERIAL_STR = None 
GSR_CONFIG_FILE = "gsr_config.json" # [NEW] Persistence file 
DEVICE_ADDRESS = None  
REAL_MANUF_STR = None
REAL_HW_STR = None
REAL_FW_STR = None
last_on_signal_time = 0.0 # [FIX] Debounce timer for headset status
ADVERTISED_NAME = "Unknown"

def battery_handler(sender, data):
    global device_battery_level
    try:
        val = int(data[0])
        device_battery_level = f"{val}%"
    except: pass
command_queue = queue.Queue()
ADVERTISED_NAME = "Unknown"
tx_seq = 1

# Filters
b_filt, a_filt = butter(4, [1.0 / (FS / 2), 100.0 / (FS / 2)], btype='band')
zi_ch1 = lfilter_zi(b_filt, a_filt)

# Biofeedback State
current_calm_score = 0
current_focus_score = 0
calm_history = collections.deque(maxlen=SCORE_SMOOTHING_WINDOW)
focus_history = collections.deque(maxlen=SCORE_SMOOTHING_WINDOW)
current_smoothing = 0.1

# --- FILES ---
BG_FILE = "Background.jpg"
SCALE_FILE = "dial_background.png"

# --- CONFIGURATION ---
VENDOR_ID = 0x1fc9
PRODUCT_ID = 0x0003
V_SOURCE = 6.371
R_REF = 83.0

# --- WINDOW SIZE ---
WIN_W = 1024
WIN_H = 800 # [FIX] Increased height for better UI spacing

# --- CALIBRATION ---
PIVOT_X = 512
PIVOT_Y = 575
SCALE_POS_X = 512
SCALE_POS_Y = 195

# --- NEEDLE GEOMETRY ---
NEEDLE_COLOR = "#882222"
NEEDLE_LENGTH_MAX = 470
NEEDLE_LENGTH_MIN = 300
NEEDLE_WIDTH = 4

# --- DIAL SETTINGS ---
ANGLE_CENTER = 90
MAX_SWEEP = 40
RESET_TARGET = 11

# --- GRAPH SETTINGS ---
GRAPH_Y = 500
GRAPH_H = 150
GRAPH_W = 700
GRAPH_X = (WIN_W - GRAPH_W) // 2
GRAPH_SCALE_V14 = 0.90  # Matches V14 scaling

# --- THRESHOLDS ---
RESET_THRESHOLD = 0.90
MOVEMENT_THRESHOLD = 0.05


# ==========================================
#           DATA LOGGER (UPDATED)
# ==========================================
# ==========================================
#           UNIFIED DATA LOGGER
# ==========================================
class SessionLogger:
    def __init__(self):
        self.file = None
        self.writer = None
        self.is_recording = False
        self.start_time = 0
        self.lock = threading.Lock()

        # Ensure directory exists
        self.folder_name = "Session_Data"
        if not os.path.exists(self.folder_name):
            try:
                os.makedirs(self.folder_name)
                print(f"[REC] Created folder: {self.folder_name}")
            except Exception as e:
                print(f"[REC] Error creating folder: {e}")
        self.recorded_samples = 0

    def start(self, filename=None, start_time=None):
        with self.lock:
            if self.is_recording: return
            if not filename:
                ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"Session_Data/GSR_Audio_{ts_str}.csv"
            
            self.filename = filename 
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.file = open(filename, mode='w', newline='')
            self.writer = csv.writer(self.file)
            header = [
                "Timestamp", "Elapsed_Sec", 
                "GSR_TA", "GSR_SetPoint", "GSR_Sens", "GSR_Motion",
                "EEG_Connected", "EEG_Headset", "EEG_Focus", "EEG_Calm",
                "EEG_Delta", "EEG_Theta", "EEG_Alpha", "EEG_Beta", "EEG_Gamma", "EEG_Raw"
            ]
            self.writer.writerow(header)
            self.is_recording = True
            
            # [FIX] Allow precise sync with Audio
            if start_time:
                self.start_time = start_time
            else:
                self.start_time = time.time()
                
            self.recorded_samples = 0
            print(f"[REC] Started: {filename} (BaseTime: {self.start_time:.4f})")

    def log_packet(self, eeg_conn, eeg_on, focus, calm, bands, raw, calculated_timestamp=None, override_gsr=None):
        with self.lock:
            if not self.is_recording: return
            
            if calculated_timestamp:
                now = calculated_timestamp
            else:
                now = time.time()
                
            if override_gsr is not None:
                gsr = override_gsr
            else:
                gsr = latest_gsr_ta
            center = latest_gsr_center
            sens = latest_gsr_sens
            mot = int(latest_gsr_motion)
            
            elapsed = now - self.start_time
            try:
                self.writer.writerow([
                    f"{now:.4f}", f"{elapsed:.4f}",
                    f"{gsr:.5f}", f"{center:.5f}", f"{sens:.3f}", mot,
                    eeg_conn, eeg_on, focus, calm,
                    f"{bands[0]:.2f}", f"{bands[1]:.2f}", f"{bands[2]:.2f}", f"{bands[3]:.2f}", f"{bands[4]:.2f}",
                    raw
                ])
                self.recorded_samples += 1
            except Exception as e:
                 print(f"[REC] Write Error: {e}")

    def log_gsr_only(self, ta, center, sens, is_motion, ble_active=False):
        """Called by GUI Thread if EEG is NOT connected"""
        if not self.is_recording: return
        if ble_active: return # If EEG is streaming, let BLE thread handle logging
        
        self.write_row(ta, center, sens, int(is_motion), 0, 0, 0, 0, [0]*5, 0)

    def write_row(self, gsr, center, sens, mot, eeg_conn, eeg_on, focus, calm, bands, raw):
        with self.lock:
            if not self.is_recording: return
            now = time.time()
            elapsed = now - self.start_time
            try:
                self.writer.writerow([
                    f"{now:.4f}", f"{elapsed:.4f}",
                    f"{gsr:.5f}", f"{center:.5f}", f"{sens:.3f}", mot,
                    eeg_conn, eeg_on, focus, calm,
                    f"{bands[0]:.2f}", f"{bands[1]:.2f}", f"{bands[2]:.2f}", f"{bands[3]:.2f}", f"{bands[4]:.2f}",
                    raw
                ])
                # [DEBUG] silenced
            except Exception as e:
                 print(f"[REC] Write Error: {e}")

    def stop(self):
        with self.lock:
            if self.file:
                self.file.close()
                self.file = None
            self.is_recording = False
            print("[REC] Saved.")


# ==========================================
#           BLE & HELPER FUNCTIONS
# ==========================================
global_session_logger = None

def log_msg(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    clean_msg = str(msg).strip()
    log_messages.append(f"[{timestamp}] {clean_msg}")
    print(clean_msg)

def encode_varint(v):
    p = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v: p.append(b | 0x80)
        else: p.append(b); break
    return p

def create_auth_packet(auth_seq=1, auth_type=1):
    serial_bytes = (REAL_SERIAL_STR if REAL_SERIAL_STR else "Unknown").encode('utf-8')
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

def fragment_packet(data, chunk_size=20):
    chunks = []
    for i in range(0, len(data), chunk_size): chunks.append(data[i:i + chunk_size])
    return chunks

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

def calculate_focus_score(bands):
    try:
        theta = bands[1]
        beta = max(1e-6, bands[3])
        if theta < 0.1 and beta < 0.1: return 0
        ratio = beta / max(1e-6, theta)
        raw_score = 50 + 50 * math.log10(ratio)
        return int(np.clip(raw_score, 0, 100))
    except: return 0

# DSP Cache for optimized processing (Ported from v10)
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

# Global band persistence
last_computed_bands = [0]*5

def notification_handler(sender, data):
    global headset_on_head, current_state, is_connected, current_calm_score, current_focus_score
    global samples_since_contact, last_packet_time, zi_ch1, eeg_buffer, last_computed_bands
    global last_on_signal_time # [FIX] Needed for debounce
    
    try:
        last_packet_time = time.time()
        if not data.startswith(b'CMSN'): return
        if timestamp_queue is not None:
            try: timestamp_queue.put_nowait(data)
            except: pass

        body = data[6:]
        # Quick Parsing (Recursive)
        def parse_payload(payload):
            global headset_on_head, last_on_signal_time # [FIX] Added last_on_signal_time
            idx = 0
            extracted = []
            while idx < len(payload):
                try: # Bounds Check
                    key = payload[idx]; idx += 1
                    if key in [0x12, 0x32]:
                        if idx >= len(payload): break
                        length = payload[idx]; idx += 1
                        if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                        if idx + length <= len(payload):
                            extracted.extend(parse_payload(payload[idx:idx + length]))
                        idx += length
                    elif key == 0x22: # Signal Data
                        if idx >= len(payload): break
                        length = payload[idx]; idx += 1
                        if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                        if idx + length <= len(payload):
                            raw_bytes = payload[idx:idx + length]
                            for i in range(0, len(raw_bytes), 3):
                                chunk = raw_bytes[i:i + 3]
                                if len(chunk) == 3: extracted.append(int.from_bytes(chunk, byteorder='big', signed=True))
                        idx += length
                    elif key == 0x18: # Status
                        if idx < len(payload):
                            val = payload[idx]; idx += 1
                            # [FIX] V10-style Debounce Logic
                            now = time.time()
                            if val == 1:
                                last_on_signal_time = now
                                if not headset_on_head: 
                                    headset_on_head = True; print("Headset: ON HEAD")
                            elif val == 2:
                                if (now - last_on_signal_time) > 1.5:
                                    if headset_on_head: 
                                        headset_on_head = False; print("Headset: OFF HEAD") 
                    elif key > 0x20: # Skip Unknown
                        if idx >= len(payload): break
                        length = payload[idx]; idx += 1
                        if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                        idx += length
                    else: idx += 1
                except IndexError: break
            return extracted
        
        extracted = parse_payload(body)
        
        # Process Signal
        if len(extracted) > 0:
            filt_chunk, zi_ch1 = lfilter(b_filt, a_filt, extracted, zi=zi_ch1)
            current_state = "LINK : STREAMING"
            
            for i, val in enumerate(filt_chunk):
                eeg_buffer.append(val)
                if headset_on_head: samples_since_contact += 1
                else: samples_since_contact = 0
                
                # Calc Bands (Rolling Window per sample, matching v10)
                bands = last_computed_bands
                
                if headset_on_head and len(eeg_buffer) >= FS:
                     # [OPTIMIZATION] Use islice to avoid copying full deque
                     recent = list(itertools.islice(eeg_buffer, len(eeg_buffer) - FS, len(eeg_buffer)))
                     new_bands = calculate_relative_bands(recent)
                     if new_bands != [0]*5:
                         last_computed_bands = new_bands
                         bands = new_bands
                     
                     current_focus_score = calculate_focus_score(bands)
                     current_calm_score = int(bands[2]) # Alpha as Calm proxy
    
                # LOGGING (UNIFIED)
                if global_session_logger and global_session_logger.is_recording:
                    # [FIX] Linear Interpolation for Smooth GSR (Upsampling 60Hz -> 250Hz)
                    # This prevents the "Blocky Data" (Staircase) effect caused by batch processing.
                    global last_gsr_interp # Need a persistent tracker
                    if 'last_gsr_interp' not in globals(): last_gsr_interp = latest_gsr_ta
                    
                    target_gsr = latest_gsr_ta
                    count = len(filt_chunk)
                    
                    # Calculate step per sample for this batch
                    # Note: We interpolate from last_log to current. 
                    if count > 0:
                        gsr_step = (target_gsr - last_gsr_interp) / count
                    else:
                        gsr_step = 0
                    
                    # [FIX] Use Wall Clock Backcasting instead of Synthetic Count
                    # This ensures correct sync even if there was a delay before the first packet.
                    now_wall = time.time()
                    chunk_duration = count / FS
                    chunk_start_time = now_wall - chunk_duration
                    
                    if global_session_logger.start_time > 0:
                         # Loop through samples
                         for i in range(count):
                             # Calculate exact time for this sample relative to NOW
                             # This aligns it with Audio Time
                             sample_time = chunk_start_time + (i / FS)
                             
                             # Smooth GSR
                             smooth_gsr = last_gsr_interp + (gsr_step * (i + 1))
                             
                             global_session_logger.log_packet(
                                True, headset_on_head, 
                                current_focus_score, current_calm_score, 
                                bands, extracted[i],
                                calculated_timestamp=sample_time,
                                override_gsr=smooth_gsr
                             )
                             global_session_logger.recorded_samples += 1
            
            # End of chunk loop: Update tracker
            if 'last_gsr_interp' in globals():
                 last_gsr_interp = target_gsr

    except Exception as e:
        print(f"[HANDLER ERR] {e}") # Print to console for debugging


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
    global REAL_MANUF_STR, REAL_HW_STR, REAL_FW_STR # [FIX] Make metadata global so GUI sees it
    STRATEGIES = [{"seq": 2, "type": 2, "delay": 1.0}, {"seq": 50, "type": 2, "delay": 1.0}, {"seq": 1, "type": 1, "delay": 2.0}]
    
    while app_running:
        try:
            current_state = "LINK : SCANNING"
            log_msg(f"[SCAN] Scanning...")
            devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
            candidate = None
            
            # Fallback or Name Match
            for d, adv in devices.values():
                if (d.name and "Brain" in d.name) or (d.name and "FC11" in d.name):
                    candidate = d; ADVERTISED_NAME = d.name; break
                if "0d740001-d26f-4dbb-95e8-a4f5c55c57a9" in adv.service_uuids:
                    candidate = d; ADVERTISED_NAME = adv.local_name if adv.local_name else d.name; break
            
            if candidate: 
                DEVICE_ADDRESS = candidate.address
                log_msg(f"[BLE] Found: {ADVERTISED_NAME} ({DEVICE_ADDRESS})")
            else: 
                log_msg("[BLE] Device Not Found. Retrying...")
                await asyncio.sleep(3.0); continue

            log_msg(f"[BLE] Connecting {DEVICE_ADDRESS}...")
            async def safe_write(cl, data, delay=0.1):
                chunks = fragment_packet(data)
                for c in chunks: await cl.write_gatt_char(UUID_WRITE, c, response=False); await asyncio.sleep(0.05)
                await asyncio.sleep(delay)

            try:
                client = BleakClient(DEVICE_ADDRESS, timeout=10.0)
                await client.connect()
                if not client.is_connected: raise Exception("Conn Fail")
                log_msg("[BLE] Connected.")
                await asyncio.sleep(0.5) # [FIX] Allow BlueZ to settle
                
                # [FIX] Robust Metadata Reader with Retry
                async def retry_read(uuid, name):
                    for attempt in range(3):
                        try:
                            await asyncio.sleep(0.5 * (attempt + 1)) # Backoff: 0.5, 1.0, 1.5
                            val = await client.read_gatt_char(uuid)
                            log_msg(f"Read {name}: OK ({val.decode('utf-8', errors='ignore').strip() if uuid != UUID_BATTERY else int(val[0])})")
                            return val
                        except Exception as e:
                            log_msg(f"Read {name} (Att {attempt+1}) Fail: {e}")
                    return None

                try:
                    # Serial (Critical)
                    s_bytes = await retry_read(UUID_SERIAL, "Serial")
                    if s_bytes: REAL_SERIAL_STR = s_bytes.decode('utf-8').strip()
                    else: raise Exception("Serial Read Exhausted")
                    
                    # Manufacturer
                    m_bytes = await retry_read(UUID_MANUFACTURER, "Manuf")
                    if m_bytes: REAL_MANUF_STR = m_bytes.decode('utf-8').strip()
                    else: REAL_MANUF_STR = "?"
                    
                    # Hardware
                    h_bytes = await retry_read(UUID_HARDWARE, "HW")
                    if h_bytes: REAL_HW_STR = h_bytes.decode('utf-8').strip()
                    else: REAL_HW_STR = "?"
                    
                    # Firmware
                    f_bytes = await retry_read(UUID_FIRMWARE, "FW")
                    if f_bytes: REAL_FW_STR = f_bytes.decode('utf-8').strip()
                    else: REAL_FW_STR = "?"
                    
                    # Battery
                    b_bytes = await retry_read(UUID_BATTERY, "Batt")
                    if b_bytes: 
                        val = int(b_bytes[0])
                        device_battery_level = f"{val}%"
                    
                except Exception as e: 
                    log_msg(f"Metadata Init Error: {e}")
                    # Don't kill connection for metadata, but Serial is usually needed
                    if REAL_SERIAL_STR == "Unknown": raise e

                current_client = client; timestamp_queue = asyncio.Queue()
                listen_task = asyncio.create_task(listener(client, timestamp_queue))

                handshake_done = False
                for idx, strat in enumerate(STRATEGIES):
                    try:
                        log_msg(f"Strategy {idx+1}: Seq {strat['seq']} Type {strat['type']}")
                        await safe_write(client, create_auth_packet(strat['seq'], strat['type']), delay=strat['delay'])
                        if not client.is_connected:
                            log_msg(f"Strategy {idx+1} Disconnected. Reconnecting...")
                            listen_task.cancel(); await client.disconnect(); await asyncio.sleep(1.0); await client.connect()
                            current_client = client; timestamp_queue = asyncio.Queue(); listen_task = asyncio.create_task(listener(client, timestamp_queue)); continue
                        
                        await safe_write(client, CMD_2_CONFIG, delay=0.2)
                        await client.start_notify(UUID_NOTIFY, notification_handler)
                        # [STABILITY] Removed Battery Notify to prevent disconnect loops
                        await safe_write(client, CMD_3_STREAM, delay=0.2)
                        await safe_write(client, CMD_4_SETUP, delay=0.2)
                        await safe_write(client, CMD_5_FINAL, delay=0.2)
                        handshake_done = True; log_msg("Handshake Success"); break
                    except Exception as e: log_msg(f"Strat {idx+1} Fail: {e}")
                
                if not handshake_done: raise Exception("Handshake Failed")

                log_msg("Streaming...")
                is_connected = True; current_state = "LINK : STREAMING"
                
                while client.is_connected and app_running:
                     while not command_queue.empty():
                          try:
                              pkt = command_queue.get_nowait()
                              await safe_write(client, pkt, delay=0.05) 
                              await asyncio.sleep(0.05) 
                          except: pass
                     await asyncio.sleep(0.1) 
                
                listen_task.cancel()
            except Exception as e: 
                log_msg(f"Error: {e}")
            finally:
                if 'client' in locals() and client.is_connected: await client.disconnect()
                if 'listen_task' in locals(): listen_task.cancel()
            await asyncio.sleep(2.0)
        except: await asyncio.sleep(2.0)

def run_ble():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bluetooth_task())

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

    def run(self):
        try:
            h = hid.device()
            h.open(VENDOR_ID, PRODUCT_ID)
            h.set_nonblocking(1) # [FIX] Non-blocking for high speed
            self.connected = True
            print("[GSR] Connected (High Speed Mode)")

            while self.running:
                # [FIX] Drain Buffer - Get latest packet
                data = None
                while True:
                    try:
                        d = h.read(64)
                        if d: data = d
                        else: break
                    except: break
                
                if not data:
                    time.sleep(0.005) # Sleep if no data to save CPU
                    continue
                
                # Check Header
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
                    
                    # [OPTIMIZATION] Update Global immediately for Logger
                    global latest_gsr_ta
                    latest_gsr_ta = ta
                
                # Yield slightly to be nice to threads, but keep polling high (~200Hz)
                time.sleep(0.002)

        except Exception as e:
            print(f"[GSR] Error: {e}")
            self.connected = False

    def stop(self):
        self.running = False

# ==========================================
#           AUDIO RECORDER (V16)
# ==========================================
class AudioRecorder:
    def __init__(self, app=None):
        self.app = app
        self.frames = []
        self.stream = None
        self.device_index = None
        self.fs = 44100
        self.gain = 1.0
        self.is_recording = False
        self.peak_level = 0.0 # For VU Meter
        self.current_mic_name = "Default"

    def probe_devices(self, target):
        """
        Finds a working mic.
        Target can be:
        - int: Specific device index (from UI selection)
        - str: Device name (from Config)
        """
        devs = sd.query_devices()
        selected_idx = None
        real_name = "NO MIC"
        
        # 1. Direct Index (User Selected in UI)
        if isinstance(target, int):
            if target < len(devs):
                try:
                    d = devs[target]
                    if d['max_input_channels'] > 0:
                        print(f"[AUDIO] Probing Index {target}: {d['name']}...")
                        with sd.InputStream(device=target, samplerate=None, channels=1) as s: s.read(10)
                        print(f"[AUDIO] Index {target} Verified.")
                        return target, d['name']
                except Exception as e:
                    print(f"[AUDIO] Index {target} Fail: {e}")
                    # Fallthrough to search by name if index failed? No, explicit index fail means fail.
                    return None, "Index Fail"

        # 2. Name Search (Config Load)
        target_name = str(target)
        print(f"[AUDIO] Probing for Name '{target_name}'...")
        
        if target_name and target_name != "Default" and target_name != "NO MIC":
            for i, d in enumerate(devs):
                if d['max_input_channels'] > 0 and (target_name in d['name'] or d['name'] in target_name):
                    try:
                        # Probe
                        with sd.InputStream(device=i, samplerate=None, channels=1) as s: s.read(10)
                        selected_idx = i
                        real_name = d['name']
                        print(f"[AUDIO] Mic Verified (Target): {real_name}")
                        break
                    except Exception as e:
                        print(f"[AUDIO] Probe Fail {d['name']}: {e}")
        
        # 3. Fallback (First working mic)
        if selected_idx is None:
            for i, d in enumerate(devs):
                if d['max_input_channels'] > 0:
                    try:
                        with sd.InputStream(device=i, samplerate=None, channels=1) as s: s.read(10)
                        selected_idx = i
                        real_name = d['name']
                        print(f"[AUDIO] Mic Verified (Fallback): {real_name}")
                        break
                    except: pass
                    
        return selected_idx, real_name

    def start(self, target_dev, gain=1.0, explicit_index=None):
        self.frames = []
        self.silent_chunks = 0
        self.total_chunks = 0
        self.gain = gain
        self.is_recording = True
        
        # 1. Resolve Device
        if explicit_index is not None:
             idx = explicit_index
             try:
                 name = sd.query_devices(idx)['name']
             except: name = "Unknown Index"
             print(f"[AUDIO] Using Explicit Index: {idx} ({name})")
        else:
             idx, name = self.probe_devices(target_dev)
        if idx is None:
            print("[AUDIO] No working microphone found.")
            self.is_recording = False
            return
            
        self.device_index = idx
        self.current_mic_name = name # Update actual name used
        
        # 2. Start Stream (SR Fallback)
        rates = [44100, 48000, 16000, None]
        # [NEW] Prioritize saved rate
        if hasattr(self.app, 'audio_mic_rate') and self.app.audio_mic_rate > 0:
            rates.insert(0, int(self.app.audio_mic_rate))
            # Remove duplicate if present
            rates = list(dict.fromkeys(rates))
            
        for sr in rates:
            try:
                self.stream = sd.InputStream(
                    device=idx, samplerate=sr, channels=1,
                    callback=self._callback, blocksize=0 # [FIX] Auto blocksize for ALSA safety
                )
                self.stream.start()
                
                # [FIX] Capture exact stream start time for Sync
                started_at = time.time()
                
                self.fs = int(self.stream.samplerate) # [FIX] Cast to int for wav.write
                self.app.audio_mic_rate = self.fs # [NEW] Save successful rate
                print(f"[AUDIO] Streaming: {name} @ {self.fs}Hz (Gain: {gain})")
                return started_at
            except Exception as e:
                print(f"[AUDIO] SR {sr} Fail: {e}")
        
        print("[AUDIO] Failed to start stream.")
        self.is_recording = False
        return None

    def _callback(self, indata, frames, time_info, status):
        # [DEBUG] Throttled print removed to prevent flooding
        # if random.random() < 0.005: print(f"[AUDIO] CB Active. Max: {np.max(np.abs(indata)):.2f}")
        
        if status: pass 
        # Apply Gain
        amplified = indata * self.gain
        self.frames.append(amplified.copy())
        
        # [DEBUG] Check for silence
        peak = np.max(np.abs(amplified))
        if peak < 0.0001: self.silent_chunks += 1
        self.total_chunks += 1


    def stop(self, filename):
        self.is_recording = False
        print("[AUDIO] Recorder.stop() called.")
        
        if self.stream:
            try:
                print("[AUDIO] calling stream.stop()...")
                self.stream.stop()
                print("[AUDIO] calling stream.close()...")
                self.stream.close()
                print("[AUDIO] stream closed.")
            except Exception as e:
                print(f"[AUDIO] Stream Close Err: {e}")
            self.stream = None
        
        if self.frames:
            print(f"[AUDIO] Processing {len(self.frames)} frames...")
            
            # Silence Report
            if self.total_chunks > 0:
                sil_pct = (self.silent_chunks / self.total_chunks) * 100.0
                print(f"[AUDIO] Stats: {self.total_chunks} chunks. {sil_pct:.1f}% Silent.")
                if sil_pct > 95.0:
                    print("[AUDIO] WARNING: Recording appears strictly silent (Sample Rate Mismatch?)")

            try:
                data = np.concatenate(self.frames, axis=0)
                # Clip
                data = np.clip(data, -1.0, 1.0)
                wav.write(filename, int(self.fs), data) # [FIX] Ensure int
                print(f"[AUDIO] Saved: {filename}")
            except Exception as e:
                print(f"[AUDIO] Write Err: {e}")
        else:
            print("[AUDIO] No data recorded.")

# ==========================================
#           SETTINGS WINDOW (TKINTER) V10-Style
# ==========================================
class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("Settings (Audio)")
        self.geometry("450x350")
        
        # Label
        tk.Label(self, text=f"Current Mic: {self.app.audio_mic_name}", fg="blue", font=("Arial", 9)).pack(pady=5)
        
        # Device List
        tk.Label(self, text="Select Microphone:").pack()
        
        devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        self.valid_devs = []
        self.index_map = {} # Map string -> index
        
        for i, d in enumerate(devs):
            if d['max_input_channels'] > 0:
                api_name = hostapis[d['hostapi']]['name']
                # Create Unique Label: [Index] Name (API)
                lbl = f"[{i}] {d['name']} ({api_name})"
                self.valid_devs.append(lbl)
                self.index_map[lbl] = i
                
        if not self.valid_devs: self.valid_devs = ["NO MIC"]
        
        # Set default selection to current match if possible
        cur_lbl = self.valid_devs[0]
        for lbl in self.valid_devs:
            if self.app.audio_mic_name in lbl:
                cur_lbl = lbl
                break
        
        self.dev_var = tk.StringVar(value=cur_lbl)
        
        opt = tk.OptionMenu(self, self.dev_var, *self.valid_devs, command=self.on_dev_select)
        opt.config(width=40)
        opt.pack(pady=5)

        # Gain
        tk.Label(self, text="Mic Gain:").pack(pady=10)
        self.gain_val = tk.DoubleVar(value=self.app.mic_gain)
        self.cached_gain = self.app.mic_gain # [FIX] Thread-safe cache
        tk.Scale(self, from_=1.0, to=10.0, orient=tk.HORIZONTAL, resolution=0.1, variable=self.gain_val, command=self.on_gain_change).pack(pady=5, fill=tk.X, padx=40)
        
        # Test Meter
        tk.Label(self, text="Test Level:").pack(pady=5)
        self.canvas = tk.Canvas(self, width=300, height=20, bg="#333")
        self.canvas.pack(pady=5)
        self.meter = self.canvas.create_rectangle(0, 0, 0, 20, fill="lime")
        
        # Monitor
        self.monitor_stream = None
        self.monitoring = True
        self.peak = 0
        self.start_monitor()
        
    def start_monitor(self):
        # [FIX] Single Control Thread Pattern
        self.restart_event = threading.Event()
        self.target_idx = None
        
        # Initial target
        lbl = self.dev_var.get()
        if lbl in self.index_map: 
             self.target_idx = self.index_map[lbl]
             # Force initial event to ensure startup
             self.restart_event.set()
        else:
             # Retry a few times if map empty?
             pass
            
        print("[SETTINGS] Starting Control Thread...")
        threading.Thread(target=self._control_loop, daemon=True).start()
        # Start UI updater
        self.update_meter()

    def _control_loop(self):
        print("[SETTINGS] Control Thread Started.")
        current_stream_idx = None
        
        while self.monitoring:
            # Check if we need to (re)start
            if self.target_idx is not None and (self.monitor_stream is None or self.target_idx != current_stream_idx or self.restart_event.is_set()):
                
                self.restart_event.clear()
                print(f"[SETTINGS] Switching to Index {self.target_idx}...")
                
                # Stop Old
                if self.monitor_stream:
                    print("[SETTINGS] Stopping old stream...")
                    try: self.monitor_stream.stop(); self.monitor_stream.close()
                    except Exception as e: print(f"[SETTINGS] Stop Err: {e}")
                    self.monitor_stream = None
                    current_stream_idx = None
                
                # Probe New
                idx = self.target_idx
                print(f"[SETTINGS] Probing Index {idx}...")
                real_idx, name = self.app.audio_recorder.probe_devices(idx)
                    
                if real_idx is not None:
                    try:
                        print(f"[SETTINGS] Starting Stream on {real_idx}...")
                        self.monitor_stream = sd.InputStream(device=real_idx, channels=1, callback=self.mon_cb, blocksize=0)
                        self.monitor_stream.start()
                        current_stream_idx = idx
                        print("[SETTINGS] Stream Started.")
                    except Exception as e:
                        print(f"[SETTINGS] Start Stream Err: {e}")
                        # Avoid rapid retry loop if broken
                        time.sleep(1.0)
                else:
                    print("[SETTINGS] Probe Failed. Retrying...")
                    time.sleep(1.0) 
                    # If initial probe failed, maybe we should unset current stream?
                    if current_stream_idx is not None:
                         current_stream_idx = None # Reset so we try again next loop
                    time.sleep(1.0) 

            time.sleep(0.2)
            
        # Cleanup on exit
        if self.monitor_stream:
            try: self.monitor_stream.stop(); self.monitor_stream.close()
            except: pass

    def mon_cb(self, indata, frames, time, status):
        try:
            # [FIX] RMS + dB Scaling (Matches v10/Standard)
            # Indata is normalized -1.0 to 1.0 (float32)
            # 1. Apply Gain
            amplified = indata * self.cached_gain
            
            # 2. Calculate RMS
            rms = np.sqrt(np.mean(amplified**2))
            
            # 3. Convert to dB (Reference 1.0)
            if rms > 0:
                db = 20 * math.log10(rms)
                # Map -50dB (Silence) to 0dB (Max) -> 0.0 to 1.0
                # clamp db to -50..0
                db_clamped = max(-50, min(0, db))
                norm = (db_clamped + 50) / 50
                self.peak = norm
            else:
                self.peak = 0
        except: pass

    def update_meter(self):
        if not self.monitoring: return
        try:
            w = min(300, self.peak * 300)
            self.canvas.coords(self.meter, 0, 0, w, 20)
            col = "lime"
            if w > 200: col = "yellow"
            if w > 280: col = "red"
            self.canvas.itemconfig(self.meter, fill=col)
        except Exception as e:
            print(f"[VU] Error: {e}")
            pass
        self.after(50, self.update_meter)

    def on_dev_select(self, val):
        if val in self.index_map:
            # print(f"[SETTINGS] User selected: {val}")
            idx = self.index_map[val]
            
            # [FIX] Do NOT call sd.query_devices here (Crash Risk)
            # Parse name from label: "[i] Name (API)"
            # Find first space after ] to start of (
            try:
                # Format: "[i] Name (API)"
                # Split by first space
                parts = val.split(' ', 1)
                if len(parts) > 1:
                     rest = parts[1] # "Name (API)"
                     # Remove last " (" bit
                     if " (" in rest:
                         name = rest.rsplit(" (", 1)[0]
                         self.app.audio_mic_name = name
                         self.app.save_gsr_config()
            except: pass

            # Signal Thread
            self.target_idx = idx
            self.restart_event.set() 

            # print("[SETTINGS] Signal sent to control thread.")

    def on_gain_change(self, val):
        f_val = float(val)
        self.app.mic_gain = f_val
        self.cached_gain = f_val # Update cache

    def destroy(self):
        self.monitoring = False
        if self.monitor_stream: 
            try: self.monitor_stream.stop(); self.monitor_stream.close()
            except: pass
        self.app.save_gsr_config()
        super().destroy()

# ==========================================
#           NOTES WINDOW (TKINTER)
# ==========================================
class NotesWindow(tk.Toplevel):
    def __init__(self, parent, app, on_start_callback=None):
        super().__init__(parent)
        self.app = app
        self.on_start = on_start_callback
        self.title("Session Notes") # Match v10
        self.geometry("500x400")    # Match v10
        
        tk.Label(self, text="Enter Session Details:", font=("Arial", 10, "bold")).pack(pady=5) # Match v10
        self.txt = tk.Text(self, font=("Arial", 10), width=50, height=15) # Match v10
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Restore existing notes or default
        if self.app.session_notes:
            self.txt.insert("1.0", self.app.session_notes)
        else:
            tpl = "Client Name: \n\nProcess Run: \n\nOther Notes: \n\n"
            self.txt.insert("1.0", tpl)
            
        self.txt.focus_set()

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.BOTTOM, pady=10) # Ensure at bottom
        
        # Make Modal - Fix Crash
        self.transient(parent)
        self.wait_visibility() # [FIX] Wait for window to be viewable
        self.grab_set()
        
        if self.on_start:
             tk.Button(btn_frame, text="Start Recording", command=self.start_session, bg="#ddffdd", height=2).pack(side=tk.LEFT, padx=10)
             tk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
        else:
             tk.Button(btn_frame, text="Save & Close", command=self.save_only).pack(pady=5)
        
    def start_session(self):
        self.app.session_notes = self.txt.get("1.0", tk.END).strip()
        self.destroy()
        if self.on_start: self.on_start()

    def save_only(self):
        self.app.session_notes = self.txt.get("1.0", tk.END).strip()
        super().destroy()

    def destroy(self):
        super().destroy()

# ==========================================
#           MAIN APP V28
# ==========================================
class ProMeterAppV28:
    def __init__(self, root):
        self.root = root
        self.root.title("GSR Meter for EK Processing")

        # State
        self.center_ta = 2.0
        self.base_sensitivity = 0.20
        self.ta_total = 0.0
        self.counting_active = False
        self.booster_level = 0
        self.last_ta_frame = 0.0
        self.is_artifact = False

        # --- GRAPH MODES ---
        self.graph_mode = "FIXED"  # Options: "FIXED" (Continuous), "NEEDLE" (V14 Original)
        self.logger = SessionLogger()
        global global_session_logger; global_session_logger = self.logger
        

        
        # Audio & Notes State
        self.audio_recorder = AudioRecorder(self) # [FIX] Pass app reference
        self.audio_mic_name = "Default" # [FIX] Name based persistence
        self.mic_gain = 1.0
        self.session_notes = ""


        # --- GRAPH HISTORY ---
        self.GRAPH_CAPACITY = 300

        # Buffer 1: Raw TA (For Fixed Graph)
        self.graph_data_raw = collections.deque(maxlen=self.GRAPH_CAPACITY)

        # Buffer 2: Needle Y-Position (For V14 Graph)
        # We store the computed Y pixel offset so it reproduces V14 exactly
        self.graph_data_v14 = collections.deque(maxlen=self.GRAPH_CAPACITY)

        self.last_graph_val = -1.0

        # Offsets
        self.p_x = PIVOT_X;
        self.p_y = PIVOT_Y
        self.s_x = SCALE_POS_X;
        self.s_y = SCALE_POS_Y

        # V14 Offset Calculation (Set Line position)
        self.graph_offset_px = (RESET_TARGET / MAX_SWEEP) * (GRAPH_H / 2)
        # Apply V14 Scale factor for exact match
        self.graph_offset_px_v14 = self.graph_offset_px * GRAPH_SCALE_V14

        # 1. Background
        try:
            raw_bg = Image.open(BG_FILE).convert("RGBA").resize((WIN_W, WIN_H), Image.LANCZOS)
            self.bg_img = ImageTk.PhotoImage(raw_bg)
        except:
            self.bg_img = None

        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.canvas = tk.Canvas(root, width=WIN_W, height=WIN_H, highlightthickness=0)
        self.canvas.pack()

        if self.bg_img:
            self.canvas.create_image(0, 0, image=self.bg_img, anchor=tk.NW)
        else:
            self.canvas.configure(bg="#888888")

        # 2. Scale
        try:
            raw_sc = Image.open(SCALE_FILE).convert("RGBA")
            self.scale_img = ImageTk.PhotoImage(raw_sc)
            self.scale_id = self.canvas.create_image(self.s_x, self.s_y, image=self.scale_img, anchor=tk.CENTER)
        except:
            self.scale_id = None

        # Bindings
        root.bind("<Key>", self.handle_keypress)
        root.bind("<space>", lambda e: self.recenter_needle())

        # Buttons
        self.canvas.tag_bind("btn_count", "<Button-1>", lambda e: self.toggle_counting())
        self.canvas.tag_bind("btn_reset", "<Button-1>", lambda e: self.reset_counter())
        self.canvas.tag_bind("btn_rec", "<Button-1>", lambda e: self.toggle_rec())
        self.canvas.tag_bind("btn_graph", "<Button-1>", lambda e: self.toggle_graph_mode())
        
        # [NEW] Bind Settings/Notes
        self.canvas.tag_bind("btn_settings", "<Button-1>", lambda e: SettingsWindow(self.root, self))
        # [REMOVED] Notes button binding

        for i in range(4): self.canvas.tag_bind(f"boost_{i}", "<Button-1>", lambda e, l=i: self.set_booster(l))

        self.sensor = GSRReader()
        self.sensor.start()
        
        self.load_gsr_config() # [NEW] Load settings on startup
        
        self.update_gui()

    def load_gsr_config(self):
        """Loads Booster and Graph settings from JSON"""
        if not os.path.exists(GSR_CONFIG_FILE): return
        try:
            with open(GSR_CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
                self.booster_level = cfg.get("booster_level", 0)
                self.graph_mode = cfg.get("graph_mode", "FIXED")
                self.audio_mic_name = cfg.get("audio_mic_name", "Default") # [FIX] Load Name
                self.audio_mic_rate = cfg.get("audio_mic_rate", 0) # [NEW] Load SR
                self.mic_gain = cfg.get("mic_gain", 1.0)
                print(f"[CONFIG] Loaded: Audio='{self.audio_mic_name}', SR={self.audio_mic_rate}, Gain={self.mic_gain}")
        except Exception as e:
            print(f"[CONFIG] Load Error: {e}")

    def save_gsr_config(self):
        """Saves current settings to JSON"""
        try:
            cfg = {
                "booster_level": self.booster_level,
                "graph_mode": self.graph_mode,
                "audio_mic_name": self.audio_mic_name, # [FIX] Save Name
                "audio_mic_rate": getattr(self, 'audio_mic_rate', 0), # [NEW] Save SR
                "mic_gain": self.mic_gain
            }
            with open(GSR_CONFIG_FILE, 'w') as f:
                json.dump(cfg, f)
            print("[CONFIG] Saved.")
        except Exception as e:
            print(f"[CONFIG] Save Error: {e}")


    def handle_keypress(self, event):
        k = event.keysym.lower();
        c = event.char
        if c in ['=', '+'] or 'plus' in k:
            self.adj_zoom(-0.05)
        elif c in ['-', '_'] or 'minus' in k:
            self.adj_zoom(0.05)
        elif c == "c":
            self.toggle_counting()
        elif c == "b":
            self.cycle_booster()
        elif c == "r":
            self.toggle_rec()
        elif c == "g":
            self.toggle_graph_mode()

    def adj_zoom(self, amount):
        self.base_sensitivity = max(0.05, self.base_sensitivity + amount)

    def toggle_counting(self):
        self.counting_active = not self.counting_active

    def reset_counter(self):
        if not self.counting_active: self.ta_total = 0.0

    def set_booster(self, level):
        self.booster_level = level

    def cycle_booster(self):
        self.booster_level = (self.booster_level + 1) % 4

    def toggle_rec(self):
        if self.logger.is_recording:
            # STOP immediately
            self.logger.stop() 
            self.update_rec_button()
        else:
            # START -> Open Notes/Setup first
            # [FIX] Workflow: REC -> Notes -> Start
            NotesWindow(self.root, self, on_start_callback=self.start_session)

    def start_session(self):
        # [FIX] Pause Monitor to prevent ALSA resource busy / fighting
        if hasattr(self, 'settings_win') and self.settings_win and self.settings_win.winfo_exists():
             print("[APP] Pausing Monitor for Recording...")
             self.settings_win.monitoring = False # Signals thread to stop stream
             time.sleep(0.5) # Wait for release

        # [FIX] Start Audio FIRST to ensure stream is ready and minimize sync drift
        # Resolve Index
        idx = None
        if hasattr(self, 'settings_win') and self.settings_win and self.settings_win.winfo_exists():
            if self.settings_win.target_idx is not None:
                idx = self.settings_win.target_idx
        
        # Start Audio
        started_at = self.audio_recorder.start(self.audio_mic_name, gain=self.mic_gain, explicit_index=idx)
        
        # Start Logger immediately after Audio returns (success)
        # This aligns the "zero" of audio (approx) with "zero" of CSV.
        if self.audio_recorder.is_recording and started_at:
            # [FIX] Pass the exact audio start time to logger for perfect sync
            self.logger.start(start_time=started_at)
            
            self.update_rec_button()
            self.session_timer = time.time()
        else:
            tk.messagebox.showerror("Error", "Audio failed to start. Recording aborted.")

    def toggle_graph_mode(self):
        # Swap between Fixed and V14
        self.graph_mode = "NEEDLE" if self.graph_mode == "FIXED" else "FIXED"

    def get_effective_zoom(self):
        if self.booster_level == 0: return self.base_sensitivity
        mult = [1.0, 0.6, 1.0, 1.4][self.booster_level]
        safe_ta = max(1.0, self.center_ta)
        return self.base_sensitivity * math.pow((2.0 / safe_ta), mult)

    def recenter_needle(self):
        if not self.sensor.connected: return
        old_ta = self.center_ta;
        new_ta = self.sensor.current_ta
        self.center_ta = new_ta
        if self.counting_active and new_ta < old_ta: self.ta_total += (old_ta - new_ta)

    def update_gui(self):
        global current_state, is_connected
        ta = self.sensor.current_ta
        is_conn = self.sensor.connected

        if self.scale_id: self.canvas.coords(self.scale_id, self.s_x, self.s_y)
        # self.canvas.delete("needle_obj"); # [OPTIMIZATION] Don't delete, update coords
        self.canvas.delete("overlay");
        self.canvas.delete("graph")

        if is_conn:
            delta = abs(ta - self.last_ta_frame)
            self.is_artifact = (delta > MOVEMENT_THRESHOLD)
            self.last_ta_frame = ta

            # --- UPDATE GLOBALS FOR BLE THREAD ---
            global latest_gsr_ta, latest_gsr_center, latest_gsr_sens, latest_gsr_motion
            latest_gsr_ta = ta
            latest_gsr_center = self.center_ta
            latest_gsr_sens = self.get_effective_zoom()
            latest_gsr_motion = self.is_artifact

            # --- LOGGING (FALLBACK) ---
            # [FIX] Watchdog: Only let BLE thread log if we are actually STREAMING.
            # If current_state is NOT streaming, we must log here even if "connected".
            ble_active = (current_state == "LINK : STREAMING")
            
            # If BLE is active, it logs. If not, we log.
            # We must ensure we don't double log. 
            # BLE thread logs "if global_session_logger.is_recording".
            # We can't easily stop BLE thread from logging if it thinks it's active.
            # But "log_gsr_only" checks "if is_connected: return".
            # Let's change "log_gsr_only" to check "if ble_active: return".
            
            # For now, let's rely on is_connected but ensure is_connected is managed.
            # Actually, better: Pass explicit flag to log_gsr_only
            self.logger.log_gsr_only(ta, self.center_ta, self.get_effective_zoom(), self.is_artifact, ble_active)

            # --- CALCULATE GEOMETRY ---
            dial_range_ta = self.get_effective_zoom()
            half_win = dial_range_ta / 2.0

            # --- NEEDLE LOGIC ---
            diff = ta - self.center_ta
            ratio = diff / half_win
            angle_v = (ratio * MAX_SWEEP) + RESET_TARGET

            if abs(angle_v) > MAX_SWEEP:
                self.recenter_needle()
                diff = ta - self.center_ta
                ratio = diff / half_win
                angle_v = (ratio * MAX_SWEEP) + RESET_TARGET

            rad = math.radians(ANGLE_CENTER + angle_v)
            tx = self.p_x + NEEDLE_LENGTH_MAX * math.cos(rad);
            ty = self.p_y - NEEDLE_LENGTH_MAX * math.sin(rad)
            bx = self.p_x + NEEDLE_LENGTH_MIN * math.cos(rad);
            by = self.p_y - NEEDLE_LENGTH_MIN * math.sin(rad)
            
            # [OPTIMIZATION] Reuse line object
            if self.canvas.find_withtag("needle_obj"):
                self.canvas.coords("needle_obj", bx, by, tx, ty)
            else:
                self.canvas.create_line(bx, by, tx, ty, width=NEEDLE_WIDTH, fill=NEEDLE_COLOR, capstyle=tk.ROUND, tags="needle_obj")

            # --- RECORD DATA TO BUFFERS ---
            if ta > 0.1 and abs(ta - self.last_graph_val) > 0.0001:
                # 1. Store Raw TA (For Fixed Graph)
                self.graph_data_raw.append(ta)

                # 2. Store V14 Needle Logic (For V14 Graph)
                # Matches V14 formula exactly: plot_y = ratio * (H/2) * Scale
                # Note: ratio = (angle_v - RESET_TARGET) / MAX_SWEEP ? No, V14 used ratio directly.
                # V14 Formula: graph_ratio = angle_from_vertical / MAX_SWEEP
                graph_ratio = angle_v / MAX_SWEEP
                # Calculate Y pixels from center
                v14_val = graph_ratio * (GRAPH_H / 2) * GRAPH_SCALE_V14
                self.graph_data_v14.append(v14_val)

                self.last_graph_val = ta

            # --- DRAW GRAPH BACKGROUND ---
            g_top = GRAPH_Y - (GRAPH_H / 2);
            g_bot = GRAPH_Y + (GRAPH_H / 2)
            self.canvas.create_rectangle(GRAPH_X, g_top, GRAPH_X + GRAPH_W, g_bot, fill="#dddddd", outline="#999",
                                         width=2, tags="graph")

            mid_y = (g_top + g_bot) / 2

            # --- RENDER SELECTED GRAPH ---

            if self.graph_mode == "FIXED":
                # === MODE A: CONTINUOUS FIXED SCALE ===
                # Set Line is at 11deg offset
                set_line_y = mid_y - self.graph_offset_px
                self.canvas.create_line(GRAPH_X, set_line_y, GRAPH_X + GRAPH_W, set_line_y, fill="#999", dash=(2, 4),
                                        tags="graph")

                graph_span = dial_range_ta * 2.0
                offset_ta = (RESET_TARGET / MAX_SWEEP) * (dial_range_ta / 2.0)
                graph_center = self.center_ta
                v_max = graph_center + (graph_span / 2.0) - offset_ta
                v_min = graph_center - (graph_span / 2.0) - offset_ta
                v_span = v_max - v_min

                if len(self.graph_data_raw) > 1:
                    step = GRAPH_W / (self.GRAPH_CAPACITY - 1)
                    points = []
                    for i, val in enumerate(self.graph_data_raw):
                        val_c = max(v_min, min(v_max, val))
                        norm = (val_c - v_min) / v_span
                        px = GRAPH_X + (i * step)
                        py = g_bot - (norm * GRAPH_H)
                        points.append(px);
                        points.append(py)
                    self.canvas.create_line(points, fill="#00aa00", width=2, tags="graph")

                    # Labels
                    lx = GRAPH_X + GRAPH_W + 8
                    self.canvas.create_text(lx, g_top + 10, text=f"{v_max:.3f}", fill="#000", font=("Arial", 9, "bold"),
                                            anchor=tk.W, tags="graph")
                    self.canvas.create_text(lx, g_bot - 10, text=f"{v_min:.3f}", fill="#000", font=("Arial", 9, "bold"),
                                            anchor=tk.W, tags="graph")
                    self.canvas.create_text(lx, set_line_y - 15, text="SPAN:", fill="#555", font=("Arial", 8),
                                            anchor=tk.W, tags="graph")
                    self.canvas.create_text(lx, set_line_y + 5, text=f"{v_span:.2g}", fill="#000",
                                            font=("Arial", 14, "bold"), anchor=tk.W, tags="graph")

            else:
                # === MODE B: V14 ORIGINAL (NEEDLE TRACKING) ===
                # Set Line matches V14 offset
                set_line_y = GRAPH_Y - self.graph_offset_px_v14
                self.canvas.create_line(GRAPH_X, set_line_y, GRAPH_X + GRAPH_W, set_line_y, fill="#999", dash=(2, 4),
                                        tags="graph")

                if len(self.graph_data_v14) > 1:
                    step = GRAPH_W / (self.GRAPH_CAPACITY - 1)
                    points = []
                    for i, val in enumerate(self.graph_data_v14):
                        # val is pixel offset from CENTER (GRAPH_Y)
                        # We need to invert it because Y grows downwards
                        py = GRAPH_Y - val
                        # Clamp
                        py = max(g_top, min(g_bot, py))
                        px = GRAPH_X + (i * step)
                        points.append(px);
                        points.append(py)
                    self.canvas.create_line(points, fill="#00aa00", width=2, tags="graph")

                    # V14 didn't have scale numbers, just the line
                    lx = GRAPH_X + GRAPH_W + 8
                    self.canvas.create_text(lx, mid_y, text="", fill="#555", font=("Arial", 10, "bold"),
                                            anchor=tk.W, tags="graph")

            # --- UI ---
            # [FIX] Moved up to make room for bottom bars
            self.draw_box(180, WIN_H - 140, "TA POSITION", f"{self.center_ta:.2f}")
            col = "#ff6600" if self.is_artifact else "black"
            lbl = "MOTION" if self.is_artifact else "INSTANT"
            self.draw_box(WIN_W - 180, WIN_H - 140, lbl, f"{ta:.3f}", col)

            # REC BUTTON (Solid Red)
            if self.logger.is_recording:
                r_col = "#cc0000"  # Nice solid red
                r_txt = "REC ON"
            else:
                r_col = "#ddd"
                r_txt = "REC OFF"

            self.canvas.create_rectangle(WIN_W / 2 - 40, 30, WIN_W / 2 + 40, 70, fill=r_col, outline="#333", width=2,
                                         tags=("overlay", "btn_rec"))
            self.canvas.create_text(WIN_W / 2, 50, text=r_txt, font=("Arial", 10, "bold"), tags=("overlay", "btn_rec"))

            # GRAPH TOGGLE BUTTON
            gx = GRAPH_X - 70
            mode_txt = "FIXED" if self.graph_mode == "FIXED" else "NEEDLE"
            self.canvas.create_rectangle(gx, GRAPH_Y - 20, gx + 60, GRAPH_Y + 20, fill="#ddd", outline="#333",
                                         tags=("overlay", "btn_graph"))
            self.canvas.create_text(gx + 30, GRAPH_Y - 5, text="VIEW:", font=("Arial", 7),
                                    tags=("overlay", "btn_graph"))
            self.canvas.create_text(gx + 30, GRAPH_Y + 8, text=mode_txt, font=("Arial", 9, "bold"),
                                    tags=("overlay", "btn_graph"))

            # Counter
            c_bg = "#ccffcc" if self.counting_active else "#ffcccc"
            c_fg = "#005500" if self.counting_active else "#550000"
            c_txt = "ON" if self.counting_active else "PAUSED"
            self.canvas.create_rectangle(20, 20, 250, 100, fill=c_bg, outline="#333", width=2,
                                         tags=("overlay", "btn_count"))
            self.canvas.create_text(135, 40, text=f"COUNTING: {c_txt}", fill=c_fg, font=("Arial", 10, "bold"),
                                    tags=("overlay", "btn_count"))
            self.canvas.create_text(135, 70, text=f"{self.ta_total:.2f}", fill="black", font=("Arial", 28, "bold"),
                                    tags=("overlay", "btn_count"))

            if not self.counting_active:
                self.canvas.create_rectangle(20, 105, 250, 135, fill="#ddd", outline="#555",
                                             tags=("overlay", "btn_reset"))
                self.canvas.create_text(135, 120, text="RESET", font=("Arial", 9, "bold"),
                                        tags=("overlay", "btn_reset"))

            bx = WIN_W - 300
            self.canvas.create_rectangle(bx, 20, WIN_W - 20, 70, fill="#eee", outline="#333", tags="overlay")
            self.canvas.create_text(bx + 140, 12, text="AUTO-BOOSTER", fill="#333", font=("Arial", 8, "bold"),
                                    tags="overlay")
            cols = ["#999", "#ee0", "#0a0", "#0aa"]
            w = 280 / 4
            for i in range(4):
                bg = cols[i] if self.booster_level == i else "#ddd"
                self.canvas.create_rectangle(bx + i * w, 20, bx + (i + 1) * w, 70, fill=bg, outline="#999",
                                             tags=("overlay", f"boost_{i}"))
                self.canvas.create_text(bx + i * w + w / 2, 45, text=["OFF", "LO", "MED", "HI"][i],
                                        font=("Arial", 10, "bold"), tags=("overlay", f"boost_{i}"))

            bw = int((1.0 / dial_range_ta) * 25)
            # [FIX] Moved Sensitivity Bar UP to separate from Status Bar
            self.canvas.create_rectangle(WIN_W / 2 - bw, WIN_H - 70, WIN_W / 2 + bw, WIN_H - 60, fill="blue",
                                         tags="overlay")
            stxt = f"SENSITIVITY: {dial_range_ta:.2f}" + (" (AUTO)" if self.booster_level > 0 else "")
            self.canvas.create_text(WIN_W / 2, WIN_H - 50, text=stxt, font=("Arial", 10, "bold"), tags="overlay")

        else:
            self.canvas.create_text(WIN_W / 2, WIN_H / 2, text="DISCONNECTED", fill="red", font=("Arial", 40, "bold"),
                                    tags="overlay")

        # --- STATUS BAR (TOP) ---
        # Link
        # [FIX] Watchdog Logic for UI
        if is_connected and time.time() - last_packet_time > 1.5:
             current_state = "LINK : TIMEOUT"
        
        lnk_col = "green" if "STREAMING" in current_state else "red"
        self.canvas.create_text(20, 2, text=current_state, fill=lnk_col, font=("Arial", 9, "bold"), anchor="nw", tags="overlay")
        
        # Headset
        hs_txt = "HEADSET: ON" if headset_on_head else "HEADSET: OFF"
        hs_col = "green" if headset_on_head else "red"
        self.canvas.create_text(200, 2, text=hs_txt, fill=hs_col, font=("Arial", 9, "bold"), anchor="nw", tags="overlay")
        
        # Battery
        self.canvas.create_text(400, 2, text=f"BATT: {device_battery_level}", fill="#555", font=("Arial", 9, "bold"), anchor="nw", tags="overlay")
        
        # [REMOVED] Scores from top bar

        # --- STATUS BAR (BOTTOM) ---
        # Name | Serial | Manuf | FW
        # dev_info requires global or retrieval. We have ADVERTISED_NAME, REAL_SERIAL_STR.
        # Manuf/FW are in 'dev_static' dict if we implement queue reading, 
        # but let's just use globals for now or placeholders.
        # [FIX] Enhanced Device Info Bar (Centered and Lowered)
        info_str = f"Name: {ADVERTISED_NAME} | SN: {REAL_SERIAL_STR if REAL_SERIAL_STR else '--'} | HW: {REAL_HW_STR if REAL_HW_STR else '--'} | FW: {REAL_FW_STR if REAL_FW_STR else '--'} | Manuf: {REAL_MANUF_STR if REAL_MANUF_STR else '--'} | Batt: {device_battery_level}"
        self.canvas.create_text(WIN_W / 2, WIN_H - 12, text=info_str, fill="#777", font=("Arial", 8), anchor="center", tags="overlay")


        # [SMOOTHNESS] Update at 50Hz (20ms) instead of 200Hz
        self.root.after(20, self.update_gui)

        # [NEW] Audio Recording Integration
        if self.logger.is_recording:
             if not self.audio_recorder.is_recording:
                 # Pass current name and gain
                 self.audio_recorder.start(self.audio_mic_name, gain=self.mic_gain)
        else:
             if self.audio_recorder.is_recording:
                 # Stop and Save
                 # Stop and Save
                 if hasattr(self.logger, 'filename') and self.logger.filename:
                      fn = self.logger.filename
                      # Force filename logic if needed (e.g. Integrated -> GSR_Audio?)
                      # But simpler to just keep same base name for pairing
                      wav_fn = fn.replace(".csv", ".wav")
                 else:
                      # [FIX] Timestamped Fallback
                      import datetime
                      ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                      wav_fn = f"Session_Data/GSR_Audio_{ts}.wav"
                 
                 self.audio_recorder.stop(wav_fn)
                 
                 # [FIX] Resume Monitor if window open
                 if hasattr(self, 'settings_win') and self.settings_win and self.settings_win.winfo_exists():
                      print("[APP] Resuming Monitor...")
                      self.settings_win.monitoring = True
                      self.settings_win.start_monitor() # Restart thread
                 
                 # Save Notes
                 if self.session_notes:
                     if 'fn' in locals() and fn:
                         txt_fn = fn.replace(".csv", "_notes.txt")
                     else:
                         # Fallback if audio-only recorded without logger file (rare but safe)
                         txt_fn = wav_fn.replace(".wav", "_notes.txt")
                         
                     with open(txt_fn, 'w') as f: f.write(self.session_notes)
                     self.session_notes = "" # Reset after save
                     print(f"[NOTES] Saved: {txt_fn}")

        # [NEW] Settings & Notes Buttons (Moved Left of Auto-Booster)
        # Auto-Booster is at: bx = WIN_W - 300 (x=724)
        # Rec Button end: x=552
        
        # Settings (Bottom Left)
        # Status Bar is at WIN_H - 12. Sensitivity at WIN_H - 50.
        # Let's put SET at x=60, y=WIN_H - 60 (Left of Sensitivity)
        bx = 60
        by = WIN_H - 60
        self.canvas.create_rectangle(bx - 30, by - 15, bx + 30, by + 15, fill="#ddd", outline="#333", tags=("overlay", "btn_settings"))
        self.canvas.create_text(bx, by, text="MIC SET", font=("Arial", 9, "bold"), tags=("overlay", "btn_settings"))
        
        # [REMOVED] Notes Button

    def draw_box(self, x, y, lbl, val, col="black"):
        self.canvas.create_rectangle(x - 90, y - 50, x + 90, y + 50, fill="#eee", outline="black", tags="overlay")
        self.canvas.create_text(x, y - 20, text=lbl, fill="#555", font=("Arial", 10, "bold"), tags="overlay")
        self.canvas.create_text(x, y + 10, text=val, fill=col, font=("Arial", 30, "bold"), tags="overlay")

    def on_close(self):
        self.save_gsr_config() # [NEW] Save on exit
        global app_running; app_running = False
        self.logger.stop()
        self.sensor.stop()
        self.root.destroy()
        
    def update_rec_button(self):
        # Helper to update REC button text/color based on state
        if self.logger.is_recording:
            self.canvas.itemconfig("btn_rec_rect", fill="red")
            self.canvas.itemconfig("btn_rec_text", text="STOP")
        else:
            self.canvas.itemconfig("btn_rec_rect", fill="#444")
            self.canvas.itemconfig("btn_rec_text", text="REC")


if __name__ == "__main__":
    # Start BLE Thread
    t_ble = threading.Thread(target=run_ble, daemon=True)
    t_ble.start()
    
    root = tk.Tk()
    app = ProMeterAppV28(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
    
    print("Main loop finished. Cleaning up...")
    app_running = False
    if t_ble.is_alive():
        print("Waiting for BLE disconnect...")
        t_ble.join(timeout=3.0)
    print("Shutdown Complete.")