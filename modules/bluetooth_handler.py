
import asyncio
import time
from datetime import datetime
from bleak import BleakClient, BleakScanner
import collections
import threading
from scipy.signal import butter, lfilter, lfilter_zi
import numpy as np

# --- UUIDS ---
UUID_BATTERY = "00002a19-0000-1000-8000-00805f9b34fb"
UUID_MODEL_NUMBER = "00002a24-0000-1000-8000-00805f9b34fb"
UUID_SERIAL = "00002a25-0000-1000-8000-00805f9b34fb"
UUID_WRITE = "0d740002-d26f-4dbb-95e8-a4f5c55c57a9"
UUID_WRITE = "0d740002-d26f-4dbb-95e8-a4f5c55c57a9"
UUID_NOTIFY = "0d740003-d26f-4dbb-95e8-a4f5c55c57a9"
UUID_DEVICE_NAME = "00002a00-0000-1000-8000-00805f9b34fb"

# --- COMMAND CONSTANTS ---
CMD_2_CONFIG = bytearray.fromhex("434d534e000c08021208080910ffc9b9f306504b4544")
CMD_3_STREAM = bytearray.fromhex("434d534e0007080312030a0103504b4544")
CMD_4_SETUP = bytearray.fromhex("434d534e0007080412030a010e504b4544")
CMD_5_FINAL = bytearray.fromhex("434d534e000608051202080d504b4544")

# --- Constants ---
FS = 250
WARMUP_SAMPLES = FS * 3

class BluetoothHandler:
    def __init__(self, raw_eeg_queue, command_queue, status_callback, log_callback):
        """
        :param raw_eeg_queue: Queue to push filtered EEG samples.
        :param command_queue: Queue to receive commands from main thread.
        :param status_callback: Function(key, value) to update main app state.
        :param log_callback: Function(msg) to log to main app.
        """
        self.raw_eeg_queue = raw_eeg_queue
        self.command_queue = command_queue
        self.status_cb = status_callback
        self.log_cb = log_callback
        self.client = None
        self.retry_count = 0
        self.max_retries = 5 # [REQ] Quit after 5 attempts
        
        self.samples_since_contact = 0
        self.last_on_signal_time = 0
        self.headset_on_head = False
        self.advertised_name = None
        
        # Filters
        self.b_filt, self.a_filt = butter(4, [1.0 / (FS / 2), 100.0 / (FS / 2)], btype='band')
        self.zi_ch1 = lfilter_zi(self.b_filt, self.a_filt)
        
        self.serial_number = None

    def encode_varint(self, v):
        p = bytearray()
        while True:
            b = v & 0x7F
            v >>= 7
            if v: p.append(b | 0x80)
            else: p.append(b); break
        return p

    def create_auth_packet(self, auth_seq=1, auth_type=1):
        if not self.serial_number: return None
        serial_bytes = self.serial_number.encode('utf-8')
        inner = bytearray([0x08, auth_type, 0x32, len(serial_bytes)]) + serial_bytes
        seq_bytes = self.encode_varint(auth_seq)
        outer = bytearray([0x08]) + seq_bytes + bytearray([0x12, len(inner)]) + inner
        pkt = bytearray.fromhex("434d534e") + bytearray([0x00, len(outer)]) + outer + bytearray.fromhex("504b4544")
        return pkt

    def log(self, msg):
        if self.log_cb: self.log_cb(msg)
        else: print(msg)

    async def run(self):
        """Main Loop for BLE"""
        self.retry_count = 0
        while True:
            # Check Retry Limit
            if self.retry_count >= self.max_retries:
                self.log(f"[BLE] Scan Limit Reached ({self.max_retries}). Quitting Scan.")
                break # Quit loop
                
            # 1. Scan
            self.retry_count += 1
            self.log(f"[BLE] Scanning for {self.advertised_name if self.advertised_name else 'Devices'} (Attempt {self.retry_count}/{self.max_retries})...")
            device = None
            try:
                # [FIX] Use return_adv=True to get advertisement data (UUIDs)
                devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
                
                # A. Try Known Name
                if self.advertised_name and self.advertised_name != "Unknown":
                    for d, adv in devices.values():
                        if d.name == self.advertised_name:
                            device = d; break
                            
                # B. Discovery
                if not device:
                    for d, adv in devices.values():
                        # Name Check
                        if d.name and ("Brain" in d.name or "FC11" in d.name or "FocusCalm" in d.name):
                            device = d
                            self.advertised_name = d.name
                            break
                        # UUID Check (Critical for some fw versions)
                        if "0d740001-d26f-4dbb-95e8-a4f5c55c57a9" in adv.service_uuids:
                            device = d
                            self.advertised_name = adv.local_name if adv.local_name else d.name
                            break
                            
            except Exception as e:
                self.log(f"[BLE] Scan Err: {e}")
                
            if not device:
                # self.log("[BLE] Device Not Found. Retrying...") # Optional to reduce spam
                await asyncio.sleep(2)
                continue
                
            self.log(f"[BLE] Found {device.name}, Connecting...")
            
            # 2. Connect
            try:
                async with BleakClient(device.address) as client:
                    self.client = client
                    self.retry_count = 0 # [REQ] Reset on success
                    self.status_cb('is_connected', True)
                    self.log(f"[BLE] Connected to {device.name}")
                    
                    # [FIX] Allow BlueZ to settle and perform Service Discovery
                    await asyncio.sleep(2.0)
                    # [FIX] Explicit Service Discovery to avoid "not performed yet" Error
                    try: await client.get_services()
                    except: pass
                    
                    # Read Serial (Required for Auth)
                    try:
                        serial_bytes = await client.read_gatt_char(UUID_SERIAL)
                        self.serial_number = serial_bytes.decode('utf-8').strip()
                        self.log(f"[BLE] Serial: {self.serial_number}")
                    except Exception as e:
                        self.log(f"[BLE] Read Serial Fail: {e}")
                        # Fallback for some fw?
                        
                    # Handshake Strategies
                    STRATEGIES = [{"seq": 2, "type": 2, "delay": 1.0}, {"seq": 50, "type": 2, "delay": 1.0}, {"seq": 1, "type": 1, "delay": 2.0}]
                    handshake_done = False
                    
                    for idx, strat in enumerate(STRATEGIES):
                        try:
                            self.log(f"[BLE] Strat {idx+1} Start...")
                            
                            # 1. Auth Packet
                            auth_pkt = self.create_auth_packet(strat['seq'], strat['type'])
                            if auth_pkt:
                                await self.safe_write(auth_pkt, delay=strat['delay'])
                            
                            # 2. Config
                            await self.safe_write(CMD_2_CONFIG, delay=0.2)
                            
                            # 3. Notify
                            await client.start_notify(UUID_NOTIFY, self.notification_handler)
                            await asyncio.sleep(0.5)
                            
                            # 4. Stream & Setup
                            await self.safe_write(CMD_3_STREAM, delay=0.2)
                            await self.safe_write(CMD_4_SETUP, delay=0.2)
                            await self.safe_write(CMD_5_FINAL, delay=0.2)
                            
                            self.log(f"[BLE] Strat {idx+1} Success!")
                            handshake_done = True; break
                        except Exception as e:
                             self.log(f"[BLE] Strat {idx+1} Fail: {e}")
                             # If fail, maybe try next strat?
                    
                    if not handshake_done:
                        self.log("[BLE] Handshake Failed. Disconnecting.")
                        await client.disconnect()
                        continue

                    # Monitoring Loop
                    while client.is_connected:
                        self.status_cb('last_packet_time', time.time())
                        
                        # Process Command Queue
                        if self.command_queue:
                            try:
                                while not self.command_queue.empty():
                                    cmd = self.command_queue.get_nowait()
                                    if isinstance(cmd, tuple):
                                        cid, pl = cmd
                                        if cid == 10:
                                            # UUID_DEVICE_NAME must be defined at top
                                            await client.write_gatt_char(UUID_DEVICE_NAME, pl)
                                            self.log(f"[BLE] Renamed to: {pl[2:].decode('utf-8')}")
                                    else:
                                        await self.safe_write(cmd)
                            except Exception as e:
                                self.log(f"[BLE] Cmd Err: {e}")
                                
                        await asyncio.sleep(0.1)
                        
                    self.log("[BLE] Disconnected.")
                    self.status_cb('is_connected', False)
                    
            except Exception as e:
                 self.log(f"[BLE] Connection Err: {e}")
                 self.status_cb('is_connected', False)
            
            self.client = None
            await asyncio.sleep(3)

    async def safe_write(self, data, delay=0.1):
        if not self.client: return
        try:
            await self.client.write_gatt_char(UUID_WRITE, data)
            await asyncio.sleep(delay)
        except Exception as e:
            self.log(f"[BLE] Write Err: {e}")

    def notification_handler(self, sender, data):
        self.status_cb('last_packet_time', time.time())
        if not data.startswith(b'CMSN'): return
        
        # pass raw data to queue? Main app used timestamp_queue for something?
        # v30 line 386: if timestamp_queue: timestamp_queue.put_nowait(data)
        # We can support timestamp_queue if needed, but simplified plan omitted it.
        # User said "Exact same", but "Bluetooth Controller".
        # We will ignore timestamp_queue if it wasn't critical (used for latency calc?).
        # v30 timestamp_queue was 'None' by default (Line 115).
        
        body = data[6:]
        try:
            new_raw = self.parse_payload(body)
            if len(new_raw) > 0:
                filt_chunk, self.zi_ch1 = lfilter(self.b_filt, self.a_filt, new_raw, zi=self.zi_ch1)
                
                # Push to Queue
                # v30: raw_eeg_queue.put_nowait(val)
                for val in filt_chunk:
                    try: self.raw_eeg_queue.put_nowait(val)
                    except: pass
                    
                    if self.headset_on_head: self.samples_since_contact += 1
                    else: self.samples_since_contact = 0
                    
        except Exception as e: pass

    def parse_payload(self, payload):
        idx = 0
        extracted = []
        while idx < len(payload):
            key = payload[idx]; idx += 1
            if key in [0x12, 0x32]:
                if idx >= len(payload): break
                length = payload[idx]; idx += 1
                if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                if idx + length <= len(payload):
                    extracted.extend(self.parse_payload(payload[idx:idx + length]))
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
                    val = payload[idx]; self.update_headset_status(val); idx += 1
            elif key == 0x08: 
                while idx < len(payload):
                     if idx >= len(payload): break
                     byte = payload[idx]; idx += 1
                     if not (byte & 0x80): break
            elif key == 0x10: idx += 1
            elif key == 0x01: 
                if idx < len(payload):
                    val = payload[idx]
                    self.status_cb('device_battery_level', f"{val}%")
                    idx += 1
            elif key > 0x20:
                if idx >= len(payload): break
                length = payload[idx]; idx += 1
                if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                idx += length
            else: idx += 1
        return extracted

    def update_headset_status(self, val):
        now = time.time()
        if val == 1:
            self.last_on_signal_time = now
            if not self.headset_on_head: 
                self.headset_on_head = True
                self.status_cb('headset_on_head', True)
                self.log("Sensor: ON HEAD")
        elif val == 2:
            if now - self.last_on_signal_time > 1.5:
                if self.headset_on_head: 
                    self.headset_on_head = False
                    self.status_cb('headset_on_head', False)
                    self.log("Sensor: OFF HEAD")
