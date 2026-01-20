import asyncio
import struct
import threading
import collections
import csv
import time
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
from matplotlib.widgets import CheckButtons, RadioButtons, Button, Slider
from scipy.signal import butter, lfilter, lfilter_zi
from bleak import BleakClient

# --- CONFIGURATION ---
DEVICE_ADDRESS = "58:94:b2:00:b4:35"
FS = 300  # Explicitly set to 300Hz as requested
MAX_WINDOW_SEC = 10
BUFFER_SIZE = int(FS * MAX_WINDOW_SEC)
HISTORY_LEN = 200
WARMUP_SAMPLES = FS * 3

# File Naming
FILENAME_MAIN = "brainwave_session"
FILENAME_DETAILED = "brainwave_detailed"

# --- UUIDS (Merged from Diagnostic) ---
UUID_BATTERY = "00002a19-0000-1000-8000-00805f9b34fb"
UUID_SERIAL = "00002a25-0000-1000-8000-00805f9b34fb"
UUID_MANUFACTURER = "00002a29-0000-1000-8000-00805f9b34fb"
UUID_MODEL_NUMBER = "00002a24-0000-1000-8000-00805f9b34fb"
UUID_FIRMWARE = "00002a26-0000-1000-8000-00805f9b34fb"
UUID_HARDWARE = "00002a27-0000-1000-8000-00805f9b34fb"

# BrainCo Specifics
UUID_WRITE = "0d740002-d26f-4dbb-95e8-a4f5c55c57a9"
UUID_NOTIFY = "0d740003-d26f-4dbb-95e8-a4f5c55c57a9"

# --- State ---
triggers_enabled = True  # Master switch for detection logic

# Commands
AUTH_COMMAND = bytearray(
    [0x43, 0x4d, 0x53, 0x4e, 0x00, 0x18, 0x08, 0x01, 0x12, 0x14, 0x08, 0x01, 0x32, 0x10, 0x30, 0x35, 0x32, 0x31, 0x65,
     0x32, 0x62, 0x64, 0x34, 0x38, 0x62, 0x62, 0x63, 0x62, 0x66, 0x36, 0x50, 0x4b, 0x45, 0x44])
START_COMMAND = bytearray(
    [0x43, 0x4d, 0x53, 0x4e, 0x00, 0x08, 0x08, 0x01, 0x32, 0x04, 0x10, 0x01, 0x18, 0x00, 0x50, 0x4b, 0x45, 0x44])
STREAM_COMMAND = bytearray(
    [0x43, 0x4d, 0x53, 0x4e, 0x00, 0x07, 0x08, 0x03, 0x12, 0x03, 0x0a, 0x01, 0x03, 0x50, 0x4b, 0x45, 0x44])

# --- STATE ---
eeg_buffer = collections.deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
bands_history = {k: collections.deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN) for k in
                 ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']}
log_messages = collections.deque(["System Ready"], maxlen=8)  # Increased log depth

# Status Flags
headset_on_head = False
is_connected = False
current_state = "DISCONNECTED"
is_recording = False
current_signal_quality = 200
last_packet_time = 0
samples_since_contact = 0
device_battery_level = "--"

# Logic Settings
current_window_sec = 1.5
current_smoothing = 0.6
event_detected = False

# Control Defaults
coincidence_window = 0.5
baseline_window_sec = 0.75
global_percent = 20.0

# Trigger Config
triggers = {
    'Delta': {'mode': 1, 'last_seen': 0, 'active': False, 'dynamic_thresh': 0},
    'Theta': {'mode': 1, 'last_seen': 0, 'active': True, 'dynamic_thresh': 0},
    'Alpha': {'mode': -1, 'last_seen': 0, 'active': True, 'dynamic_thresh': 0},
    'Beta': {'mode': 1, 'last_seen': 0, 'active': True, 'dynamic_thresh': 0},
    'Gamma': {'mode': 1, 'last_seen': 0, 'active': False, 'dynamic_thresh': 0}
}

# CSV Handles
csv_file = None
csv_writer = None
csv_file_detailed = None
csv_writer_detailed = None
recording_start_time = None

# Filters
# 24-bit data @ 300Hz
b_filt, a_filt = butter(4, [1.0 / (FS / 2), 50.0 / (FS / 2)], btype='band')
zi_ch1 = lfilter_zi(b_filt, a_filt)


# --- FUNCTIONS ---
def calculate_focus_score(bands):
    """
    Calculates a 0-100 Focus Score based on Beta/Theta ratio.
    bands list order: [Delta, Theta, Alpha, Beta, Gamma]
    """
    try:
        theta = bands[1]
        beta = max(1e-6, bands[3])  # Prevent divide by zero

        # Avoid noise: if signals are essentially zero, return 0
        if theta < 0.1 and beta < 0.1:
            return 0

        # Beta / Theta Ratio
        ratio = beta / max(1e-6, theta)

        # Convert to 0-100 Scale (Logarithmic)
        # Ratio 1.0 -> Score 50
        # Ratio 10.0 -> Score 100
        # Ratio 0.1 -> Score 0
        raw_score = 50 + 50 * math.log10(ratio)

        # Clamp between 0 and 100
        return int(np.clip(raw_score, 0, 100))
    except:
        return 0

def calculate_relative_bands(clean_data):
    try:
        fft_vals = np.abs(np.fft.rfft(clean_data))
        fft_freq = np.fft.rfftfreq(len(clean_data), 1.0 / FS)

        def pwr(low, high):
            idx = np.where((fft_freq >= low) & (fft_freq <= high))
            return np.mean(fft_vals[idx]) if len(idx[0]) > 0 else 0

        delta = pwr(0.5, 4)
        theta = pwr(4, 8)
        alpha = pwr(8, 13)
        beta = pwr(13, 30)
        gamma = pwr(30, 50)

        total = delta + theta + alpha + beta + gamma
        if total == 0: return [0] * 5

        return [
            (delta / total) * 100,
            (theta / total) * 100,
            (alpha / total) * 100,
            (beta / total) * 100,
            (gamma / total) * 100
        ]
    except:
        return [0] * 5


def log_msg(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Clean up newline chars from BLE reads
    clean_msg = str(msg).strip()
    log_messages.append(f"[{timestamp}] {clean_msg}")


def notification_handler(sender, data):
    global headset_on_head, current_state, is_recording, current_signal_quality
    global samples_since_contact, last_packet_time, zi_ch1

    last_packet_time = time.time()
    if not data.startswith(b'CMSN'): return

    body = data[8:]
    idx = 0
    new_raw = []

    try:
        while idx < len(body):
            if idx >= len(body): break
            key = body[idx]
            idx += 1

            # --- EEG DATA (0x22) ---
            if key == 0x22:
                if idx >= len(body): break
                length = body[idx]
                idx += 1
                if length & 0x80:
                    length = (length & 0x7F) | (body[idx] << 7)
                    idx += 1
                if idx + length > len(body): break

                # 24-bit Unpacking (Big Endian)
                raw_bytes = body[idx: idx + length]
                for i in range(0, len(raw_bytes), 3):
                    chunk = raw_bytes[i:i + 3]
                    if len(chunk) == 3:
                        val = int.from_bytes(chunk, byteorder='big', signed=True)
                        new_raw.append(val)

                if len(new_raw) > 0:
                    # Apply continuous filter
                    filt_chunk, zi_ch1 = lfilter(b_filt, a_filt, new_raw, zi=zi_ch1)
                    eeg_buffer.extend(filt_chunk)

                    current_state = "STREAMING"
                    if headset_on_head:
                        samples_since_contact += len(new_raw)
                    else:
                        samples_since_contact = 0
                idx += length

            # --- SIGNAL QUALITY (0x08) ---
            elif key == 0x08:
                if idx >= len(body): break
                val = body[idx]
                # FIX: Just accept whatever the device says.
                # Usually: 0=Good, 200=Off, Other=Noise level
                current_signal_quality = val
                idx += 1

            # --- HEADSET STATUS (0x10) ---
            elif key == 0x10:
                if idx >= len(body): break
                val = body[idx]
                if val == 1:
                    if not headset_on_head:
                        headset_on_head = True
                        log_msg("Sensor: ON HEAD")
                else:
                    if headset_on_head:
                        headset_on_head = False
                        log_msg("Sensor: OFF HEAD")
                idx += 1
            else:
                idx += 1

        if current_signal_quality == 200 and headset_on_head:
            # Fallback check
            pass

    except Exception:
        pass

    # --- RECORDING LOGIC ---
    if is_recording and csv_writer:
        ts = datetime.now().strftime('%H:%M:%S.%f')
        bands = [0] * 5
        calm_score = 0
        is_warmed_up = samples_since_contact > WARMUP_SAMPLES

        if headset_on_head and is_warmed_up and len(eeg_buffer) > FS * 2:
            bands = calculate_relative_bands(list(eeg_buffer)[-int(FS * 2):])
            try:
                a, t, b = bands[2], bands[1], max(1e-6, bands[3])
                ratio = t / (a + b)
                raw = 30 * math.log10(ratio) + 50
                calm_score = int(np.clip(raw, 0, 100))
            except:
                pass

        evt_flag = 1 if event_detected else 0
        # Log summarized row
        row = [ts, current_state, int(headset_on_head), evt_flag, current_signal_quality] + bands + list(new_raw[:50])
        csv_writer.writerow(row)

        if csv_writer_detailed and len(new_raw) > 0:
            try:
                # Log detailed raw packets
                raw_data = new_raw
                num = len(raw_data)
                packet_end = datetime.now()
                interval = 1.0 / FS
                rel_start = (packet_end - recording_start_time).total_seconds()

                rows = []
                for i in range(num):
                    offset = (num - 1 - i) * interval
                    abs_ts = (packet_end - timedelta(seconds=offset)).strftime('%H:%M:%S.%f')
                    rel_ts = f"{rel_start - offset:.4f}"

                    rows.append([
                        abs_ts, rel_ts,
                        raw_data[i],  # Raw 24-bit
                        current_signal_quality, int(headset_on_head),
                        f"{bands[0]:.2f}", f"{bands[1]:.2f}", f"{bands[2]:.2f}", f"{bands[3]:.2f}", f"{bands[4]:.2f}",
                        calm_score
                    ])
                csv_writer_detailed.writerows(rows)
            except:
                pass


async def bluetooth_task():
    global is_connected, current_state, last_packet_time, device_battery_level
    log_msg(f"Searching {DEVICE_ADDRESS}...")

    while True:
        try:
            async with BleakClient(DEVICE_ADDRESS, timeout=20.0) as client:
                log_msg("Connected!")
                is_connected = True
                current_state = "DIAGNOSTIC"
                t0 = time.time()

                # --- 1. IDENTITY & DIAGNOSTICS ---
                log_msg("Reading Device Info...")

                # Helper to safely read chars
                async def safe_read(uuid, name):
                    try:
                        val = await client.read_gatt_char(uuid)
                        res = val.decode('utf-8')
                        log_msg(f"{name}: {res}")
                        return res
                    except:
                        log_msg(f"{name}: [Fail]")
                        return "?"

                await safe_read(UUID_MANUFACTURER, "Mfg")
                await safe_read(UUID_MODEL_NUMBER, "Model")
                await safe_read(UUID_SERIAL, "Serial")
                await safe_read(UUID_FIRMWARE, "FW")

                # BATTERY
                try:
                    val = await client.read_gatt_char(UUID_BATTERY)
                    pct = int.from_bytes(val, byteorder='little')
                    device_battery_level = f"{pct}%"
                    log_msg(f"Battery: {pct}%")
                except:
                    device_battery_level = "??"
                    log_msg("Battery: Failed")

                # --- 2. AUTHENTICATION ---
                log_msg("Authenticating...")
                await client.write_gatt_char(UUID_WRITE, AUTH_COMMAND[0:20], response=False)
                await asyncio.sleep(0.05)
                await client.write_gatt_char(UUID_WRITE, AUTH_COMMAND[20:], response=False)

                log_msg("Waiting for Unlock...")
                await asyncio.sleep(1.5)

                # --- 3. START STREAM ---
                log_msg("Starting Stream...")
                current_state = "HANDSHAKE"
                last_packet_time = time.time()

                await client.start_notify(UUID_NOTIFY, notification_handler)
                await client.write_gatt_char(UUID_WRITE, START_COMMAND, response=False)
                await asyncio.sleep(0.5)
                await client.write_gatt_char(UUID_WRITE, STREAM_COMMAND, response=False)

                log_msg("Stream Active")

                # Keep connection alive
                while client.is_connected:
                    await asyncio.sleep(1)

                log_msg("Disconnected")
                is_connected = False
                current_state = "DISCONNECTED"

        except Exception as e:
            is_connected = False
            current_state = "RETRYING"
            # Optional: log the specific error to console if needed
            # print(f"Connection Error: {e}")
            await asyncio.sleep(2.0)


def run_ble():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bluetooth_task())


if __name__ == "__main__":
    t = threading.Thread(target=run_ble, daemon=True)
    t.start()

    # --- GUI SETUP ---
    # Box 1: Trigger Settings (Top Right Control)
    rect_trig = [0.65, 0.84, 0.3, 0.14]
    # Box 2: Signal Processing (Bottom Right Control)
    rect_proc = [0.65, 0.25, 0.3, 0.10]

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(left=0.05, right=0.60, top=0.88, bottom=0.1)

    colors = {'Delta': 'blue', 'Theta': 'green', 'Alpha': 'orange', 'Beta': 'red', 'Gamma': 'purple'}
    lines = {b: ax.plot([], [], lw=2, label=b, color=colors[b])[0] for b in bands_history}

    mean_lines = {b: ax.axhline(y=0, color=colors[b], linestyle='--', alpha=0.3, linewidth=1) for b in bands_history}
    thresh_lines = {b: ax.axhline(y=0, color=colors[b], linestyle=':', alpha=0.8, linewidth=1.5) for b in bands_history}

    ax.set_xlim(0, HISTORY_LEN)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Brainwave Event Detector (24-bit @ {FS}Hz)")
    ax.set_ylabel("Relative Power (%)")

    # Status Texts
    txt_conn = ax.text(0.15, 1.05, "DISCONNECTED", transform=ax.transAxes, ha="center", fontsize=10, fontweight='bold',
                       color='gray')
    txt_wear = ax.text(0.40, 1.05, "HEADSET OFF", transform=ax.transAxes, ha="center", fontsize=10, fontweight='bold',
                       color='red')
    txt_batt = ax.text(0.60, 1.05, "BATT: --", transform=ax.transAxes, ha="center", fontsize=10, fontweight='bold',
                       color='gray')
    txt_sig = ax.text(0.85, 1.05, "SIGNAL --", transform=ax.transAxes, ha="center", fontsize=10, fontweight='bold',
                      color='gray')

    rec_text = ax.text(0.95, 0.95, "● REC", transform=ax.transAxes, color='red', fontsize=12, fontweight='bold',
                       visible=False)
    txt_score = ax.text(0.50, 0.95, "Score: --", transform=ax.transAxes, ha="center", fontsize=14, fontweight='bold',
                        color='gray')

    event_text = ax.text(0.5, 0.5, "⚠️ EVENT DETECTED", transform=ax.transAxes,
                         ha="center", va="center", fontsize=30, fontweight='bold',
                         color='red', bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'), visible=False)


    # --- DRAW BOXES ---
    def create_panel(rect, title):
        ax_p = plt.axes(rect)
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        ax_p.set_facecolor('#e0e0e0')
        ax_p.text(0.05, 0.95, title, transform=ax_p.transAxes, ha='left', va='top', fontsize=9, fontweight='bold',
                  color='#333')
        for spine in ax_p.spines.values(): spine.set_edgecolor('gray')
        return ax_p


    ax_trig_bg = create_panel(rect_trig, "Trigger Settings")
    ax_proc_bg = create_panel(rect_proc, "Signal Processing")

    # --- CONTROLS ---
    ui_refs = {}

    # 1. Master Sliders
    ax_coin = plt.axes([rect_trig[0] + 0.05, rect_trig[1] + 0.08, rect_trig[2] - 0.07, 0.03])
    ui_refs['slide_win'] = Slider(ax_coin, 'Coincidence (s)', 0.1, 2.0, valinit=coincidence_window, color='gray')
    ui_refs['slide_win'].label.set_size(8)
    ui_refs['slide_win'].valtext.set_fontsize(8)

    # 1. Master Trigger Checkbox (NEW)
    # Position it inside the Trigger Settings box
    ax_master_trig = plt.axes([rect_trig[0] + 0.15, rect_trig[1] + 0.105, 0.14, 0.03])

    # Match the gray background of the panel so it blends in
    ax_master_trig.set_facecolor('#e0e0e0')

    # Remove the black box border (spines) around the checkbox area
    for spine in ax_master_trig.spines.values():
        spine.set_visible(False)

    ui_refs['chk_master'] = CheckButtons(ax_master_trig, ['Active'], [True],
                                         label_props={'fontweight': ['bold'], 'fontsize': [9]})
    # Style the checkbox
    if hasattr(ui_refs['chk_master'], 'rectangles'):
        ui_refs['chk_master'].rectangles[0].set_facecolor('red')
        ui_refs['chk_master'].rectangles[0].set_edgecolor('gray')


    def toggle_master_triggers(label):
        global triggers_enabled
        triggers_enabled = not triggers_enabled
        # Force hide lines immediately if turned off
        if not triggers_enabled:
            for b in bands_history:
                mean_lines[b].set_visible(False)
                thresh_lines[b].set_visible(False)
            event_text.set_visible(False)
        plt.draw()


    ui_refs['chk_master'].on_clicked(toggle_master_triggers)
    def update_coin(val):
        global coincidence_window;
        coincidence_window = val


    ui_refs['slide_win'].on_changed(update_coin)

    ax_base = plt.axes([rect_trig[0] + 0.05, rect_trig[1] + 0.05, rect_trig[2] - 0.07, 0.03])
    ui_refs['slide_base'] = Slider(ax_base, 'Baseline Win (s)', 0.1, 2.0, valinit=baseline_window_sec,
                                   color='lightblue')
    ui_refs['slide_base'].label.set_size(8)
    ui_refs['slide_base'].valtext.set_fontsize(8)


    def update_base(val):
        global baseline_window_sec;
        baseline_window_sec = val


    ui_refs['slide_base'].on_changed(update_base)

    ax_thresh = plt.axes([rect_trig[0] + 0.05, rect_trig[1] + 0.02, rect_trig[2] - 0.07, 0.03])
    ui_refs['slide_thresh'] = Slider(ax_thresh, 'Threshold (%)', 0, 50, valinit=global_percent, color='gold')
    ui_refs['slide_thresh'].label.set_size(8)
    ui_refs['slide_thresh'].valtext.set_fontsize(8)


    def update_thresh(val):
        global global_percent;
        global_percent = val


    ui_refs['slide_thresh'].on_changed(update_thresh)

    # 2. Re-Calibrate & Record
    ax_relock = plt.axes([0.65, 0.77, 0.14, 0.05])
    ui_refs['btn_relock'] = Button(ax_relock, 'Re-Calibrate', color='lightblue', hovercolor='cyan')

    # 3. Window Slider
    ax_win = plt.axes([rect_proc[0] + 0.05, rect_proc[1] + 0.05, rect_proc[2] - 0.07, 0.03])
    ui_refs['slide_window'] = Slider(ax_win, 'Sample (s)', 0.5, 4.0, valinit=current_window_sec, color='cyan')
    ui_refs['slide_window'].label.set_size(8)
    ui_refs['slide_window'].valtext.set_fontsize(8)


    def update_window(val):
        global current_window_sec;
        current_window_sec = val


    ui_refs['slide_window'].on_changed(update_window)

    # 4. Smoothing Slider
    ax_smooth = plt.axes([rect_proc[0] + 0.05, rect_proc[1] + 0.02, rect_proc[2] - 0.07, 0.03])
    ui_refs['slide_smooth'] = Slider(ax_smooth, 'Responsiveness', 0.1, 1.0, valinit=current_smoothing, color='magenta')
    ui_refs['slide_smooth'].label.set_text('Smoothing')
    ui_refs['slide_smooth'].label.set_size(8)
    ui_refs['slide_smooth'].valtext.set_fontsize(8)


    def update_smooth(val):
        global current_smoothing;
        current_smoothing = val


    ui_refs['slide_smooth'].on_changed(update_smooth)


    def relock(e):
        log_msg("Buffers Cleared")
        eeg_buffer.clear()


    ui_refs['btn_relock'].on_clicked(relock)

    ax_rec = plt.axes([0.81, 0.77, 0.14, 0.05])
    ui_refs['btn_rec'] = Button(ax_rec, 'Record', color='lightgreen', hovercolor='lime')


    def toggle_rec(e):
        global is_recording, csv_file, csv_writer, csv_file_detailed, csv_writer_detailed, recording_start_time
        if not is_recording:
            ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname_main = f"{FILENAME_MAIN}_{ts_str}.csv"
            fname_detail = f"{FILENAME_DETAILED}_{ts_str}.csv"
            try:
                # Main Log
                csv_file = open(fname_main, 'w', newline='')
                csv_writer = csv.writer(csv_file)
                header = ["Timestamp", "State", "Headset_On", "Event_Detected", "Signal_Quality", "Delta", "Theta",
                          "Alpha", "Beta", "Gamma"] + [f"Raw_{i}" for i in range(50)]
                csv_writer.writerow(header)

                # Detailed Log
                csv_file_detailed = open(fname_detail, 'w', newline='')
                csv_writer_detailed = csv.writer(csv_file_detailed)
                header_detail = ["Abs_Time", "Rel_Time", "Raw_24bit", "Quality", "Headset", "Delta", "Theta", "Alpha",
                                 "Beta", "Gamma", "Calm_Score"]
                csv_writer_detailed.writerow(header_detail)

                recording_start_time = datetime.now()
                is_recording = True

                ui_refs['btn_rec'].label.set_text("Stop")
                ui_refs['btn_rec'].color = 'salmon'
                rec_text.set_visible(True)
                ax.set_title(f"Recording: {ts_str}")
                log_msg(f"Started: {ts_str}")
            except Exception as ex:
                log_msg(f"Error: {ex}")
        else:
            is_recording = False
            if csv_file: csv_file.close()
            if csv_file_detailed: csv_file_detailed.close()

            ui_refs['btn_rec'].label.set_text("Record")
            ui_refs['btn_rec'].color = 'lightgreen'
            rec_text.set_visible(False)
            ax.set_title(f"Brainwave Event Detector (24-bit @ {FS}Hz)")
            log_msg("Recording Saved")


    ui_refs['btn_rec'].on_clicked(toggle_rec)

    # 5. Band Rows
    start_y = 0.70
    row_height = 0.08

    for i, band in enumerate(bands_history.keys()):
        y_pos = start_y - (i * row_height)
        ax_chk = plt.axes([0.65, y_pos, 0.20, 0.05])
        try:
            init_state = triggers[band]['active']
            chk = CheckButtons(ax_chk, [band], [init_state],
                               label_props={'color': [colors[band]], 'fontweight': ['bold']})
            if hasattr(chk, 'rectangles') and len(chk.rectangles) > 0:
                chk.rectangles[0].set_facecolor(colors[band])
        except:
            chk = CheckButtons(ax_chk, [band], [True])
        ui_refs[f'chk_{band}'] = chk

        ax_dir = plt.axes([0.86, y_pos, 0.08, 0.05])
        lbl = "↑" if triggers[band]['mode'] == 1 else "↓"
        btn_dir = Button(ax_dir, lbl, color='white', hovercolor='0.9')
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

    # 6. Log Window
    ax_log = plt.axes([0.65, 0.09, 0.30, 0.15])
    ax_log.axis('off')
    log_text = ax_log.text(0, 1, "", va="top", fontsize=8, fontfamily='monospace')


    def update(frame):
        # Update Log
        log_text.set_text("\n".join(log_messages))

        # Update Connection Status UI
        if is_connected and (time.time() - last_packet_time > 2.0) and current_state == "STREAMING":
            txt_conn.set_text("LINK: STALLED")
            txt_conn.set_color("red")
        elif is_connected:
            txt_conn.set_text(f"LINK: {current_state}")
            txt_conn.set_color("black")
        else:
            txt_conn.set_text("LINK: DISCONNECTED")
            txt_conn.set_color("gray")

        # Update Battery
        txt_batt.set_text(f"BATT: {device_battery_level}")

        if headset_on_head:
            txt_wear.set_text("HEADSET: ON")
            txt_wear.set_color("green")
        else:
            txt_wear.set_text("HEADSET: OFF")
            txt_wear.set_color("red")

        if current_signal_quality == 0:
            txt_sig.set_text(f"SIG: PERFECT ({current_signal_quality})")
            txt_sig.set_color("green")
        elif current_signal_quality < 50:
            txt_sig.set_text(f"SIG: OK ({current_signal_quality})")
            txt_sig.set_color("blue")
        elif current_signal_quality == 200:
            txt_sig.set_text("SIG: OFF HEAD")
            txt_sig.set_color("red")
        else:
            txt_sig.set_text(f"SIG: NOISY ({current_signal_quality})")
            txt_sig.set_color("orange")

        if is_recording: rec_text.set_alpha(1.0 if frame % 10 < 5 else 0.3)

        # Update Plots
        req = int(FS * current_window_sec)
        if len(eeg_buffer) >= req and headset_on_head:
            snap = list(eeg_buffer)[-req:]
            new_bands = calculate_relative_bands(snap)
            now = time.time()

            hist_points = int(baseline_window_sec * 20)
            hist_points = max(5, hist_points)

            bands_checked = 0
            recent_triggers = 0
            vis_vals = []

            # ... inside the loop where you process bands ...

            for i, k in enumerate(bands_history):
                # (Existing smoothing logic)
                val = new_bands[i]
                if len(bands_history[k]) > 0:
                    prev = bands_history[k][-1]
                    val = prev * (1 - current_smoothing) + val * current_smoothing
                bands_history[k].append(val)

                # --- UPDATED VISUALIZATION LOGIC ---
                if triggers[k]['active']:
                    lines[k].set_data(range(len(bands_history[k])), bands_history[k])
                    vis_vals.extend(list(bands_history[k]))

                    # >>> MASTER SWITCH CHECK <<<
                    if triggers_enabled:
                        bands_checked += 1

                        # --- DYNAMIC THRESHOLD CALCULATION ---
                        if len(bands_history[k]) > hist_points:
                            recent = list(bands_history[k])[-hist_points:]
                            mean = np.mean(recent)

                            # Update Mean Line
                            mean_lines[k].set_ydata([mean])
                            mean_lines[k].set_visible(True)

                            mode = triggers[k]['mode']
                            percent = global_percent / 100.0
                            if mode == 1:
                                thresh = mean * (1.0 + percent)
                            else:
                                thresh = max(0, mean * (1.0 - percent))

                            triggers[k]['dynamic_thresh'] = thresh

                            # Update Threshold Line
                            thresh_lines[k].set_ydata([thresh])
                            thresh_lines[k].set_visible(True)
                            vis_vals.append(thresh)

                            if (mode == 1 and val > thresh) or (mode == -1 and val < thresh):
                                triggers[k]['last_seen'] = now

                        if now - triggers[k]['last_seen'] < coincidence_window:
                            recent_triggers += 1
                    else:
                        # If triggers disabled, hide the helper lines
                        mean_lines[k].set_visible(False)
                        thresh_lines[k].set_visible(False)
                else:
                    # If band disabled, hide everything
                    lines[k].set_visible(False)
                    mean_lines[k].set_visible(False)
                    thresh_lines[k].set_visible(False)

            # Update Score
            try:
                # 1. Calculate Both
                focus_val = calculate_focus_score(new_bands)

                # Re-calculate Calm (using your existing logic)
                a_pwr = bands_history['Alpha'][-1]
                t_pwr = bands_history['Theta'][-1]
                b_pwr = max(1e-6, bands_history['Beta'][-1])
                calm_ratio = t_pwr / (a_pwr + b_pwr)
                calm_raw = 30 * math.log10(calm_ratio) + 50
                calm_val = int(np.clip(calm_raw, 0, 100))

                # 2. Concatenate Strings
                display_text = f"Focus: {focus_val} | Calm: {calm_val}"
                txt_score.set_text(display_text)

                # 3. Dynamic Coloring Logic
                # If Focus is high, go Green. If Focus is low, go Blue/Orange.
                if focus_val > 60:
                    txt_score.set_color('green')  # High Focus
                elif focus_val < 30:
                    txt_score.set_color('blue')  # Low Focus (likely relaxed)
                else:
                    txt_score.set_color('#333333')  # Neutral (Dark Gray)

            except:
                pass


            if bands_checked > 0 and recent_triggers == bands_checked:
                event_detected = True
                event_text.set_visible(True)
            else:
                event_detected = False
                event_text.set_visible(False)

            if vis_vals:
                mx = max(vis_vals)
                ax.set_ylim(0, mx * 1.1)

        return list(lines.values()) + list(thresh_lines.values()) + list(mean_lines.values()) + [txt_conn, txt_wear,
                                                                                                 txt_sig, rec_text,
                                                                                                 event_text, txt_score,
                                                                                                 txt_batt]


    try:
        ani = animation.FuncAnimation(fig, update, interval=50)
        plt.show()
    finally:
        if csv_file: csv_file.close()
        if csv_file_detailed: csv_file_detailed.close()