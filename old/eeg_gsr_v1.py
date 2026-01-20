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
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from bleak import BleakClient, BleakScanner

# ==========================================
#              CONFIGURATION
# ==========================================

# --- GSR SETTINGS ---
VENDOR_ID = 0x1fc9
PRODUCT_ID = 0x0003
V_SOURCE = 6.371
R_REF = 83.0

# --- EEG SETTINGS (From v10-main.py) ---
FS = 250
EEG_WINDOW_SEC = 1.0  # Window for FFT calculation
EEG_BUFFER_SIZE = int(FS * 2)  # Keep 2 seconds buffer

# UUIDs
UUID_WRITE = "0d740002-d26f-4dbb-95e8-a4f5c55c57a9"
UUID_NOTIFY = "0d740003-d26f-4dbb-95e8-a4f5c55c57a9"
UUID_SERIAL = "00002a25-0000-1000-8000-00805f9b34fb"

# Handshake Commands
CMD_2_CONFIG = bytearray.fromhex("434d534e000c08021208080910ffc9b9f306504b4544")
CMD_3_STREAM = bytearray.fromhex("434d534e0007080312030a0103504b4544")
CMD_4_SETUP = bytearray.fromhex("434d534e0007080412030a010e504b4544")
CMD_5_FINAL = bytearray.fromhex("434d534e000608051202080d504b4544")

# --- FILES ---
BG_FILE = "Background.jpg"
SCALE_FILE = "dial_background.png"

# --- VISUALS ---
WIN_W = 1024
WIN_H = 768
PIVOT_X = 512;
PIVOT_Y = 575
SCALE_POS_X = 512;
SCALE_POS_Y = 195
NEEDLE_COLOR = "#882222"
NEEDLE_LENGTH_MAX = 470;
NEEDLE_LENGTH_MIN = 300;
NEEDLE_WIDTH = 4
ANGLE_CENTER = 90;
MAX_SWEEP = 40;
RESET_TARGET = 11

# Graph
GRAPH_Y = 500;
GRAPH_H = 150;
GRAPH_W = 700;
GRAPH_X = (WIN_W - GRAPH_W) // 2
GRAPH_SCALE = 0.90
GRAPH_OFFSET_PX = (RESET_TARGET / MAX_SWEEP) * (GRAPH_H / 2) * GRAPH_SCALE
RESET_THRESHOLD = 0.90
MOVEMENT_THRESHOLD = 0.05


# ==========================================
#           EEG PROCESSING LOGIC
# ==========================================
class EEGProcessor:
    def __init__(self):
        # Filters (Butterworth Bandpass 1-100Hz)
        self.b, self.a = butter(4, [1.0 / (FS / 2), 100.0 / (FS / 2)], btype='band')
        self.zi = lfilter_zi(self.b, self.a)

        # Buffers
        self.raw_buffer = collections.deque(maxlen=EEG_BUFFER_SIZE)

        # FFT Cache
        self.dsp_cache = {'len': 0, 'window': None, 'freq': None, 'idx': {}}

        # State Output
        self.bands = [0.0] * 5  # Delta, Theta, Alpha, Beta, Gamma
        self.focus = 0
        self.calm = 0
        self.connected = False
        self.headset_on = False

    def process_packet(self, raw_data):
        # 1. Filter Data
        filt_data, self.zi = lfilter(self.b, self.a, raw_data, zi=self.zi)
        self.raw_buffer.extend(filt_data)

        # 2. Check Buffer Size
        if len(self.raw_buffer) >= int(FS * EEG_WINDOW_SEC):
            # Get latest window
            clean_data = list(self.raw_buffer)[-int(FS * EEG_WINDOW_SEC):]
            self.calc_fft(np.array(clean_data))

    def calc_fft(self, clean_data):
        n = len(clean_data)
        if n == 0: return

        # Init Cache if needed
        if self.dsp_cache['len'] != n:
            self.dsp_cache['len'] = n
            self.dsp_cache['window'] = np.hanning(n)
            freqs = np.fft.rfftfreq(n, 1.0 / FS)
            self.dsp_cache['freq'] = freqs

            def get_idx(low, high): return np.where((freqs >= low) & (freqs < high))

            self.dsp_cache['idx'] = {
                'd': get_idx(0.5, 4), 't': get_idx(4, 8),
                'a': get_idx(8, 13), 'b': get_idx(13, 30), 'g': get_idx(30, 50)
            }

        # Perform FFT
        windowed = clean_data * self.dsp_cache['window']
        fft_vals = np.abs(np.fft.rfft(windowed))

        def pwr(key):
            idxs = self.dsp_cache['idx'][key]
            if len(idxs[0]) == 0: return 0
            return np.sum(fft_vals[idxs])

        d = pwr('d');
        t = pwr('t');
        a = pwr('a');
        b = pwr('b');
        g = pwr('g')
        total = d + t + a + b + g

        if total > 0:
            self.bands = [(d / total) * 100, (t / total) * 100, (a / total) * 100, (b / total) * 100, (g / total) * 100]

            # Simple Scores (Matches your v10 logic)
            # Calm = ratio of Alpha to high freq
            try:
                raw_c = 30 * math.log10(self.bands[2] / (self.bands[3] + max(1e-6, self.bands[4]))) + 50
                self.calm = int(np.clip(raw_c, 0, 100))
            except:
                self.calm = 0

            # Focus = Beta / Theta
            try:
                ratio = self.bands[3] / max(1e-6, self.bands[1])
                raw_f = 50 + 50 * math.log10(ratio)
                self.focus = int(np.clip(raw_f, 0, 100))
            except:
                self.focus = 0


# ==========================================
#           EEG BLUETOOTH THREAD
# ==========================================
class EEGReader(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.processor = EEGProcessor()
        self.loop = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.ble_task())

    def stop(self):
        # Stop not fully implemented for async loop, usually just killing app works
        pass

    async def ble_task(self):
        print("[EEG] Scanning...")
        while True:
            try:
                dev = await BleakScanner.find_device_by_filter(
                    lambda d, ad: "Brain" in (d.name or "") or "FC11" in (d.name or "") or "Muse" in (d.name or ""),
                    timeout=5.0
                )

                if not dev:
                    print("[EEG] No device found, retrying...")
                    await asyncio.sleep(2.0)
                    continue

                print(f"[EEG] Found {dev.name}, Connecting...")
                async with BleakClient(dev.address) as client:
                    if not client.is_connected: continue
                    print("[EEG] Connected!")
                    self.processor.connected = True

                    # --- AUTH/HANDSHAKE (From v10) ---
                    # 1. Get Serial
                    try:
                        serial = await client.read_gatt_char(UUID_SERIAL)
                        s_str = serial.decode('utf-8')
                        auth_pkt = self.create_auth_packet(s_str)
                    except:
                        auth_pkt = self.create_auth_packet("Unknown")  # Fallback

                    await self.safe_write(client, auth_pkt)
                    await asyncio.sleep(0.5)
                    await self.safe_write(client, CMD_2_CONFIG)

                    # Start Stream
                    await client.start_notify(UUID_NOTIFY, self.notification_handler)
                    await self.safe_write(client, CMD_3_STREAM)
                    await self.safe_write(client, CMD_4_SETUP)
                    await self.safe_write(client, CMD_5_FINAL)

                    print("[EEG] Streaming...")

                    while client.is_connected:
                        await asyncio.sleep(1.0)

                    self.processor.connected = False
                    print("[EEG] Disconnected.")

            except Exception as e:
                print(f"[EEG] Error: {e}")
                self.processor.connected = False
                await asyncio.sleep(2.0)

    def notification_handler(self, sender, data):
        # Basic parsing logic from v10 (simplified for speed)
        # We look for Key 0x22 (Data) or 0x18 (Status)
        if not data.startswith(b'CMSN'): return

        payload = data[6:]
        idx = 0
        extracted = []

        while idx < len(payload):
            key = payload[idx];
            idx += 1
            if key == 0x22:  # Data
                if idx >= len(payload): break
                length = payload[idx];
                idx += 1
                if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                if idx + length <= len(payload):
                    raw_bytes = payload[idx:idx + length]
                    for i in range(0, len(raw_bytes), 3):
                        chunk = raw_bytes[i:i + 3]
                        if len(chunk) == 3:
                            val = int.from_bytes(chunk, byteorder='big', signed=True)
                            extracted.append(val)
                idx += length
            elif key == 0x18:  # Status
                if idx < len(payload):
                    val = payload[idx]
                    self.processor.headset_on = (val == 1)
                    idx += 1
            else:
                # Skip unknown keys
                if idx >= len(payload): break
                # Crude skip logic for variable length keys
                # (For robustness, assume most keys < 0x20 are 1 byte, > 0x20 have length)
                if key > 0x20:
                    length = payload[idx];
                    idx += 1
                    if length & 0x80: length = (length & 0x7F) | (payload[idx] << 7); idx += 1
                    idx += length

        if extracted:
            self.processor.process_packet(extracted)

    def create_auth_packet(self, serial_str):
        # Re-implementation of v10 auth
        def encode_varint(v):
            p = bytearray()
            while True:
                b = v & 0x7F;
                v >>= 7
                if v:
                    p.append(b | 0x80)
                else:
                    p.append(b); break
            return p

        serial_bytes = serial_str.encode('utf-8')
        inner = bytearray([0x08, 0x02, 0x32, len(serial_bytes)]) + serial_bytes
        seq_bytes = encode_varint(2)  # Seq 2
        outer = bytearray([0x08]) + seq_bytes + bytearray([0x12, len(inner)]) + inner
        return bytearray.fromhex("434d534e") + bytearray([0x00, len(outer)]) + outer + bytearray.fromhex("504b4544")

    async def safe_write(self, client, data):
        # Fragment to 20 bytes
        for i in range(0, len(data), 20):
            chunk = data[i:i + 20]
            await client.write_gatt_char(UUID_WRITE, chunk, response=False)
            await asyncio.sleep(0.05)


# ==========================================
#           GSR READER (HID)
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
            h.set_nonblocking(0)
            self.connected = True

            while self.running:
                try:
                    data = h.read(64, timeout_ms=5)
                except:
                    self.connected = False; time.sleep(0.1); continue

                if data and len(data) >= 4 and data[0] == 0x01:
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
        except:
            self.connected = False

    def stop(self):
        self.running = False


# ==========================================
#           DATA LOGGER
# ==========================================
class SessionLogger:
    def __init__(self):
        self.file = None;
        self.writer = None;
        self.is_recording = False;
        self.start_time = 0

    def start(self):
        if self.is_recording: return
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f"BioSession_{ts}.csv"
        try:
            self.file = open(self.filename, mode='w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                "Timestamp_Unix", "Elapsed_Sec",
                "GSR_TA", "GSR_Instant", "GSR_Artifact",
                "EEG_Connected", "EEG_OnHead", "EEG_Focus", "EEG_Calm",
                "EEG_Delta", "EEG_Theta", "EEG_Alpha", "EEG_Beta", "EEG_Gamma"
            ])
            self.start_time = time.time()
            self.is_recording = True
            print(f"[REC] Started: {self.filename}")
        except Exception as e:
            print(f"[REC] Error: {e}")

    def log(self, gsr_data, eeg_proc):
        if not self.is_recording: return
        now = time.time();
        elapsed = now - self.start_time

        self.writer.writerow([
            f"{now:.4f}", f"{elapsed:.4f}",
            f"{gsr_data[0]:.4f}", f"{gsr_data[1]:.4f}", int(gsr_data[2]),
            int(eeg_proc.connected), int(eeg_proc.headset_on),
            eeg_proc.focus, eeg_proc.calm,
            f"{eeg_proc.bands[0]:.2f}", f"{eeg_proc.bands[1]:.2f}", f"{eeg_proc.bands[2]:.2f}",
            f"{eeg_proc.bands[3]:.2f}", f"{eeg_proc.bands[4]:.2f}"
        ])

    def stop(self):
        if self.file: self.file.close()
        self.is_recording = False
        print("[REC] Saved.")


# ==========================================
#           MAIN GUI APP
# ==========================================
class ProMeterAppV20:
    def __init__(self, root):
        self.root = root
        self.root.title("GSR + EEG Biofeedback System V20")

        self.center_ta = 2.0;
        self.base_sensitivity = 0.20
        self.ta_total = 0.0;
        self.counting_active = False;
        self.booster_level = 0
        self.last_ta_frame = 0.0;
        self.is_artifact = False;
        self.rec_blink = False
        self.graph_data = collections.deque([0] * 900, maxlen=900)

        self.p_x = PIVOT_X;
        self.p_y = PIVOT_Y;
        self.s_x = SCALE_POS_X;
        self.s_y = SCALE_POS_Y
        self.logger = SessionLogger()

        # Init Graphics
        try:
            raw = Image.open(BG_FILE).convert("RGBA").resize((WIN_W, WIN_H), Image.LANCZOS)
            self.bg_img = ImageTk.PhotoImage(raw)
        except:
            self.bg_img = None
        self.canvas = tk.Canvas(root, width=WIN_W, height=WIN_H, highlightthickness=0)
        self.canvas.pack()
        if self.bg_img: self.canvas.create_image(0, 0, image=self.bg_img, anchor=tk.NW)

        try:
            s_raw = Image.open(SCALE_FILE).convert("RGBA")
            self.scale_img = ImageTk.PhotoImage(s_raw)
            self.scale_id = self.canvas.create_image(self.s_x, self.s_y, image=self.scale_img, anchor=tk.CENTER)
        except:
            self.scale_id = None

        # Bindings
        root.bind("<Key>", self.key_handler)
        root.bind("<space>", lambda e: self.recenter())
        self.canvas.tag_bind("btn_count", "<Button-1>", lambda e: self.toggle_count())
        self.canvas.tag_bind("btn_reset", "<Button-1>", lambda e: self.reset_count())
        self.canvas.tag_bind("btn_rec", "<Button-1>", lambda e: self.toggle_rec())
        for i in range(4): self.canvas.tag_bind(f"boost_{i}", "<Button-1>",
                                                lambda e, l=i: setattr(self, 'booster_level', l))

        # Start Threads
        self.gsr = GSRReader();
        self.gsr.start()
        self.eeg = EEGReader();
        self.eeg.start()

        self.update_gui()

    def key_handler(self, e):
        k = e.keysym.lower();
        c = e.char
        if c in ['=', '+']:
            self.base_sensitivity = max(0.05, self.base_sensitivity - 0.05)
        elif c in ['-', '_']:
            self.base_sensitivity += 0.05
        elif c == 'c':
            self.toggle_count()
        elif c == 'r':
            self.toggle_rec()

    def toggle_count(self):
        self.counting_active = not self.counting_active

    def reset_count(self):
        if not self.counting_active: self.ta_total = 0.0

    def toggle_rec(self):
        if self.logger.is_recording:
            self.logger.stop()
        else:
            self.logger.start()

    def recenter(self):
        if not self.gsr.connected: return
        old = self.center_ta;
        new = self.gsr.current_ta
        self.center_ta = new
        if self.counting_active and new < old: self.ta_total += (old - new)

    def get_zoom(self):
        s = [0.20, 0.20 * (2.0 / max(1, self.center_ta)) ** 0.6,
             0.20 * (2.0 / max(1, self.center_ta)),
             0.20 * (2.0 / max(1, self.center_ta)) ** 1.4]
        return self.base_sensitivity * (s[self.booster_level] / 0.20)

    def update_gui(self):
        ta = self.gsr.current_ta

        # Artifact
        delta = abs(ta - self.last_ta_frame)
        self.is_artifact = (delta > MOVEMENT_THRESHOLD)
        self.last_ta_frame = ta

        # Logging
        self.logger.log((self.center_ta, ta, self.is_artifact), self.eeg.processor)

        # Needle Math
        zoom = self.get_zoom();
        half = zoom / 2.0
        diff = ta - self.center_ta
        ratio = diff / half
        angle_v = (ratio * MAX_SWEEP) + RESET_TARGET

        if abs(angle_v) > MAX_SWEEP:
            self.recenter()
            angle_v = RESET_TARGET  # Snap to set point

        rad = math.radians(ANGLE_CENTER + angle_v)

        # Graph
        plot_y = (angle_v / MAX_SWEEP) * (GRAPH_H / 2) * GRAPH_SCALE
        self.graph_data.append(plot_y)

        # Draw
        self.canvas.delete("needle_obj");
        self.canvas.delete("overlay");
        self.canvas.delete("graph")

        # Needle
        tx = self.p_x + NEEDLE_LENGTH_MAX * math.cos(rad);
        ty = self.p_y - NEEDLE_LENGTH_MAX * math.sin(rad)
        bx = self.p_x + NEEDLE_LENGTH_MIN * math.cos(rad);
        by = self.p_y - NEEDLE_LENGTH_MIN * math.sin(rad)
        self.canvas.create_line(bx, by, tx, ty, width=NEEDLE_WIDTH, fill=NEEDLE_COLOR, capstyle=tk.ROUND,
                                tags="needle_obj")

        # Graph Box
        gy = GRAPH_Y;
        gh = GRAPH_H;
        gx = GRAPH_X;
        gw = GRAPH_W
        self.canvas.create_rectangle(gx, gy - gh / 2, gx + gw, gy + gh / 2, fill="#dddddd", outline="#999", width=2,
                                     tags="graph")
        sl_y = gy - GRAPH_OFFSET_PX
        self.canvas.create_line(gx, sl_y, gx + gw, sl_y, fill="#999", dash=(2, 4), tags="graph")

        pts = []
        step = gw / len(self.graph_data)
        for i, val in enumerate(self.graph_data):
            pts.append(gx + i * step);
            pts.append(max(gy - gh / 2, min(gy + gh / 2, gy - val)))
        if len(pts) >= 4: self.canvas.create_line(pts, fill="#00aa00", width=2, tags="graph")

        # UI Text
        self.draw_box(180, WIN_H - 100, "TA", f"{self.center_ta:.2f}")
        col = "#ff6600" if self.is_artifact else "black"
        lbl = "MOTION" if self.is_artifact else "INSTANT"
        self.draw_box(WIN_W - 180, WIN_H - 100, lbl, f"{ta:.3f}", col)

        # Counter
        c_col = "#ccffcc" if self.counting_active else "#ffcccc"
        self.canvas.create_rectangle(20, 20, 250, 100, fill=c_col, outline="#333", width=2,
                                     tags=("overlay", "btn_count"))
        self.canvas.create_text(135, 70, text=f"{self.ta_total:.2f}", font=("Arial", 28, "bold"),
                                tags=("overlay", "btn_count"))
        if not self.counting_active:
            self.canvas.create_rectangle(20, 105, 250, 135, fill="#ddd", outline="#555", tags=("overlay", "btn_reset"))
            self.canvas.create_text(135, 120, text="RESET", font=("Arial", 9, "bold"), tags=("overlay", "btn_reset"))

        # Rec Button
        if self.logger.is_recording:
            self.rec_blink = not self.rec_blink
            r_col = "#ff8888" if self.rec_blink else "red"
        else:
            r_col = "#ccc"
        self.canvas.create_rectangle(WIN_W / 2 - 40, 30, WIN_W / 2 + 40, 70, fill=r_col, outline="#333", width=2,
                                     tags=("overlay", "btn_rec"))
        self.canvas.create_text(WIN_W / 2, 50, text="REC", font=("Arial", 12, "bold"), tags=("overlay", "btn_rec"))

        # Booster Radio
        bx = WIN_W - 300
        self.canvas.create_rectangle(bx, 20, WIN_W - 20, 70, fill="#eee", outline="#333", tags="overlay")
        cols = ["#999", "#ee0", "#0a0", "#0aa"]
        w = 280 / 4
        for i in range(4):
            bg = cols[i] if self.booster_level == i else "#ddd"
            self.canvas.create_rectangle(bx + i * w, 20, bx + (i + 1) * w, 70, fill=bg, outline="#999",
                                         tags=("overlay", f"boost_{i}"))
            self.canvas.create_text(bx + i * w + w / 2, 45, text=["OFF", "LO", "MED", "HI"][i],
                                    font=("Arial", 10, "bold"), tags=("overlay", f"boost_{i}"))

        # EEG Status
        e_stat = "EEG: CONN" if self.eeg.processor.connected else "EEG: --"
        e_col = "#00aa00" if self.eeg.processor.connected else "#555"
        self.canvas.create_text(WIN_W - 60, WIN_H - 20, text=e_stat, fill=e_col, font=("Arial", 10, "bold"),
                                tags="overlay")

        self.root.after(5, self.update_gui)

    def draw_box(self, x, y, lbl, val, col="black"):
        self.canvas.create_rectangle(x - 90, y - 50, x + 90, y + 50, fill="#eee", outline="black", tags="overlay")
        self.canvas.create_text(x, y - 20, text=lbl, fill="#555", font=("Arial", 10, "bold"), tags="overlay")
        self.canvas.create_text(x, y + 10, text=val, fill=col, font=("Arial", 30, "bold"), tags="overlay")

    def on_close(self):
        if self.logger.is_recording: self.logger.stop()
        self.gsr.stop()
        # EEG stop not fully implemented due to async loop nature, just exit
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = ProMeterAppV20(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()