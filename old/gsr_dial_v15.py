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
WIN_H = 768

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
class SessionLogger:
    def __init__(self):
        self.file = None
        self.writer = None
        self.is_recording = False
        self.start_time = 0

        # Ensure directory exists
        self.folder_name = "Session_Data"
        if not os.path.exists(self.folder_name):
            try:
                os.makedirs(self.folder_name)
                print(f"[REC] Created folder: {self.folder_name}")
            except Exception as e:
                print(f"[REC] Error creating folder: {e}")

    def start(self):
        if self.is_recording: return
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"GSR_Session_{ts}.csv"
        # Full path
        filepath = os.path.join(self.folder_name, filename)

        try:
            self.file = open(filepath, mode='w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(["Timestamp", "Elapsed_Sec", "TA_Instant", "TA_SetPoint", "Sensitivity", "Is_Motion"])
            self.start_time = time.time()
            self.is_recording = True
            print(f"[REC] Started: {filepath}")
        except Exception as e:
            print(f"[REC] Error opening file: {e}")

    def log(self, ta, center_ta, sensitivity, is_motion):
        if not self.is_recording: return
        now = time.time()
        elapsed = now - self.start_time
        try:
            self.writer.writerow([
                f"{now:.4f}",
                f"{elapsed:.4f}",
                f"{ta:.5f}",
                f"{center_ta:.5f}",
                f"{sensitivity:.3f}",
                int(is_motion)
            ])
        except:
            pass

    def stop(self):
        if self.file:
            self.file.close()
            self.file = None
        self.is_recording = False
        print("[REC] Saved.")


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
            h.set_nonblocking(0)
            self.connected = True

            while self.running:
                try:
                    data = h.read(64, timeout_ms=20)
                except:
                    self.connected = False;
                    time.sleep(0.1);
                    continue

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
                elif not data:
                    pass
        except:
            self.connected = False

    def stop(self):
        self.running = False


# ==========================================
#           MAIN APP V28
# ==========================================
class ProMeterAppV28:
    def __init__(self, root):
        self.root = root
        self.root.title("GSR Meter V28 (Session_Data & V14 Graph)")

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

        for i in range(4): self.canvas.tag_bind(f"boost_{i}", "<Button-1>", lambda e, l=i: self.set_booster(l))

        self.sensor = GSRReader()
        self.sensor.start()
        self.update_gui()

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
            self.logger.stop()
        else:
            self.logger.start()

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
        ta = self.sensor.current_ta
        is_conn = self.sensor.connected

        if self.scale_id: self.canvas.coords(self.scale_id, self.s_x, self.s_y)
        self.canvas.delete("needle_obj");
        self.canvas.delete("overlay");
        self.canvas.delete("graph")

        if is_conn:
            delta = abs(ta - self.last_ta_frame)
            self.is_artifact = (delta > MOVEMENT_THRESHOLD)
            self.last_ta_frame = ta

            # --- LOGGING ---
            self.logger.log(ta, self.center_ta, self.get_effective_zoom(), self.is_artifact)

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
            self.canvas.create_line(bx, by, tx, ty, width=NEEDLE_WIDTH, fill=NEEDLE_COLOR, capstyle=tk.ROUND,
                                    tags="needle_obj")

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
            self.draw_box(180, WIN_H - 100, "TA POSITION", f"{self.center_ta:.2f}")
            col = "#ff6600" if self.is_artifact else "black"
            lbl = "MOTION" if self.is_artifact else "INSTANT"
            self.draw_box(WIN_W - 180, WIN_H - 100, lbl, f"{ta:.3f}", col)

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
            self.canvas.create_rectangle(WIN_W / 2 - bw, WIN_H - 40, WIN_W / 2 + bw, WIN_H - 30, fill="blue",
                                         tags="overlay")
            stxt = f"SENSITIVITY: {dial_range_ta:.2f}" + (" (AUTO)" if self.booster_level > 0 else "")
            self.canvas.create_text(WIN_W / 2, WIN_H - 20, text=stxt, font=("Arial", 10, "bold"), tags="overlay")

        else:
            self.canvas.create_text(WIN_W / 2, WIN_H / 2, text="DISCONNECTED", fill="red", font=("Arial", 40, "bold"),
                                    tags="overlay")

        self.root.after(5, self.update_gui)

    def draw_box(self, x, y, lbl, val, col="black"):
        self.canvas.create_rectangle(x - 90, y - 50, x + 90, y + 50, fill="#eee", outline="black", tags="overlay")
        self.canvas.create_text(x, y - 20, text=lbl, fill="#555", font=("Arial", 10, "bold"), tags="overlay")
        self.canvas.create_text(x, y + 10, text=val, fill=col, font=("Arial", 30, "bold"), tags="overlay")

    def on_close(self):
        self.logger.stop()
        self.sensor.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProMeterAppV28(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()