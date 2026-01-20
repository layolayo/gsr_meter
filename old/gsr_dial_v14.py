import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import hid
import threading
import time
import math
import sys
import os
import collections

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

# --- NEEDLE GEOMETRY (UPDATED) ---
NEEDLE_COLOR = "#882222"
NEEDLE_LENGTH_MAX = 470  # User Request
NEEDLE_LENGTH_MIN = 300  # User Request
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
GRAPH_SCALE = 0.90
GRAPH_OFFSET_PX = (RESET_TARGET / MAX_SWEEP) * (GRAPH_H / 2) * GRAPH_SCALE

# --- RESET THRESHOLD ---
RESET_THRESHOLD = 0.90

# --- ARTIFACT DETECTION SETTINGS (NEW) ---
# If TA changes more than this in 5ms, it's a body movement
MOVEMENT_THRESHOLD = 0.05


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
            h.set_nonblocking(0)  # Blocking mode for max speed
            self.connected = True

            while self.running:
                try:
                    data = h.read(64, timeout_ms=5)
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


class ProMeterAppV17:
    def __init__(self, root):
        self.root = root
        self.root.title("GSR Professional Meter V1")

        # State
        self.center_ta = 2.0
        self.base_sensitivity = 0.20
        self.ta_total = 0.0
        self.counting_active = False
        self.booster_level = 0

        # Artifact State
        self.last_ta_frame = 0.0
        self.is_artifact = False

        # Graph History - Reduced to 900 to speed up visual scrolling (~10%)
        self.graph_data = collections.deque([0] * 500, maxlen=500)

        self.p_x = PIVOT_X;
        self.p_y = PIVOT_Y
        self.s_x = SCALE_POS_X;
        self.s_y = SCALE_POS_Y

        # 1. Background
        try:
            raw_bg = Image.open(BG_FILE).convert("RGBA")
            raw_bg = raw_bg.resize((WIN_W, WIN_H), Image.LANCZOS)
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
            self.scale_pil = Image.open(SCALE_FILE).convert("RGBA")
            self.scale_img = ImageTk.PhotoImage(self.scale_pil)
            self.scale_id = self.canvas.create_image(self.s_x, self.s_y, image=self.scale_img, anchor=tk.CENTER)
        except:
            self.scale_img = None
            self.scale_id = None

        # --- CONTROLS ---
        root.bind("<Key>", self.handle_keypress)
        root.bind("<space>", lambda e: self.recenter_needle())

        self.canvas.tag_bind("btn_count", "<Button-1>", lambda e: self.toggle_counting())
        self.canvas.tag_bind("btn_reset", "<Button-1>", lambda e: self.reset_counter())
        self.canvas.tag_bind("boost_0", "<Button-1>", lambda e: self.set_booster(0))
        self.canvas.tag_bind("boost_1", "<Button-1>", lambda e: self.set_booster(1))
        self.canvas.tag_bind("boost_2", "<Button-1>", lambda e: self.set_booster(2))
        self.canvas.tag_bind("boost_3", "<Button-1>", lambda e: self.set_booster(3))

        self.sensor = GSRReader()
        self.sensor.start()
        self.update_gui()

    def handle_keypress(self, event):
        k = event.keysym.lower()
        c = event.char
        if c in ['=', '+'] or 'plus' in k:
            self.adj_zoom(-0.05)
        elif c in ['-', '_'] or 'minus' in k:
            self.adj_zoom(0.05)
        elif c == "c":
            self.toggle_counting()
        elif c == "b":
            self.cycle_booster()

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

    def get_effective_zoom(self):
        if self.booster_level == 0: return self.base_sensitivity
        if self.booster_level == 1:
            strength = 0.6
        elif self.booster_level == 2:
            strength = 1.0
        elif self.booster_level == 3:
            strength = 1.4
        safe_ta = max(1.0, self.center_ta)
        return self.base_sensitivity * math.pow((2.0 / safe_ta), strength)

    def recenter_needle(self):
        if not self.sensor.connected: return
        old_ta = self.center_ta
        new_ta = self.sensor.current_ta
        self.center_ta = new_ta
        if self.counting_active:
            if new_ta < old_ta: self.ta_total += (old_ta - new_ta)

    def update_gui(self):
        ta = self.sensor.current_ta
        is_conn = self.sensor.connected

        if self.scale_id: self.canvas.coords(self.scale_id, self.s_x, self.s_y)
        self.canvas.delete("needle_obj")
        self.canvas.delete("overlay")
        self.canvas.delete("graph")

        if is_conn:
            # --- ARTIFACT DETECTION ---
            # Calculate instantaneous velocity (Change since last frame)
            delta = abs(ta - self.last_ta_frame)
            if delta > MOVEMENT_THRESHOLD:
                self.is_artifact = True
            else:
                self.is_artifact = False

            self.last_ta_frame = ta

            eff_zoom = self.get_effective_zoom()
            half_win = eff_zoom / 2.0

            diff = ta - self.center_ta
            ratio = diff / half_win

            angle_from_vertical = (ratio * MAX_SWEEP) + RESET_TARGET

            if abs(angle_from_vertical) > MAX_SWEEP:
                self.recenter_needle()
                diff = ta - self.center_ta
                ratio = diff / half_win
                angle_from_vertical = (ratio * MAX_SWEEP) + RESET_TARGET

            angle_rad = math.radians(ANGLE_CENTER + angle_from_vertical)

            # --- GRAPH UPDATE ---
            graph_ratio = angle_from_vertical / MAX_SWEEP
            plot_y = graph_ratio * (GRAPH_H / 2) * GRAPH_SCALE
            self.graph_data.append(plot_y)

            # --- NEEDLE ---
            tip_x = self.p_x + NEEDLE_LENGTH_MAX * math.cos(angle_rad)
            tip_y = self.p_y - NEEDLE_LENGTH_MAX * math.sin(angle_rad)
            base_x = self.p_x + NEEDLE_LENGTH_MIN * math.cos(angle_rad)
            base_y = self.p_y - NEEDLE_LENGTH_MIN * math.sin(angle_rad)

            self.canvas.create_line(base_x, base_y, tip_x, tip_y,
                                    width=NEEDLE_WIDTH, fill=NEEDLE_COLOR, capstyle=tk.ROUND, tags="needle_obj")

            # --- GRAPH BOX ---
            g_top = GRAPH_Y - (GRAPH_H / 2)
            g_bot = GRAPH_Y + (GRAPH_H / 2)
            self.canvas.create_rectangle(GRAPH_X, g_top, GRAPH_X + GRAPH_W, g_bot,
                                         fill="#dddddd", outline="#999", width=2, tags="graph")
            set_line_y = GRAPH_Y - GRAPH_OFFSET_PX
            self.canvas.create_line(GRAPH_X, set_line_y, GRAPH_X + GRAPH_W, set_line_y, fill="#999", dash=(2, 4),
                                    tags="graph")

            # Plot line
            points = []
            step = GRAPH_W / len(self.graph_data)
            for i, val in enumerate(self.graph_data):
                px = GRAPH_X + (i * step)
                py = GRAPH_Y - val
                py = max(g_top, min(g_bot, py))
                points.append(px)
                points.append(py)

            if len(points) >= 4:
                # If artifact, maybe change graph color? For now we just stick to green
                self.canvas.create_line(points, fill="#00aa00", width=2, tags="graph")

            # --- UI READOUTS ---
            self.draw_readout(180, WIN_H - 100, "TA POSITION", f"{self.center_ta:.2f}", "#000000")

            # INST Readout Color Logic
            inst_col = "#000000"
            inst_label = "INSTANT TA"
            if self.is_artifact:
                inst_col = "#ff6600"  # Orange warning color
                inst_label = "BODY MOTION"

            self.draw_readout(WIN_W - 180, WIN_H - 100, inst_label, f"{ta:.3f}", inst_col)

            # --- COUNTER ---
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
                self.canvas.create_rectangle(20, 105, 250, 135, fill="#dddddd", outline="#555", width=1,
                                             tags=("overlay", "btn_reset"))
                self.canvas.create_text(135, 120, text="RESET", fill="#333", font=("Arial", 9, "bold"),
                                        tags=("overlay", "btn_reset"))

            # --- BOOSTER ---
            bx_start = WIN_W - 300
            bx_end = WIN_W - 20
            self.canvas.create_rectangle(bx_start, 20, bx_end, 70, fill="#eeeeee", outline="#333", width=2,
                                         tags="overlay")
            self.canvas.create_text(bx_start + 140, 12, text="AUTO-BOOSTER", fill="#333", font=("Arial", 8, "bold"),
                                    tags="overlay")
            seg_w = (bx_end - bx_start) / 4
            labels = ["OFF", "LO", "MED", "HI"]
            cols = ["#999999", "#eeee00", "#00aa00", "#00aaaa"]
            for i in range(4):
                x1 = bx_start + (i * seg_w)
                x2 = x1 + seg_w
                is_active = (self.booster_level == i)
                bg_col = cols[i] if is_active else "#dddddd"
                txt_col = "black" if is_active else "#777777"
                font_w = "bold" if is_active else "normal"
                tag_name = f"boost_{i}"
                self.canvas.create_rectangle(x1, 20, x2, 70, fill=bg_col, outline="#999", width=1,
                                             tags=("overlay", tag_name))
                self.canvas.create_text((x1 + x2) / 2, 45, text=labels[i], fill=txt_col, font=("Arial", 10, font_w),
                                        tags=("overlay", tag_name))

            # --- SENSITIVITY ---
            bar_w = int((1.0 / eff_zoom) * 25)
            self.canvas.create_rectangle(WIN_W / 2 - bar_w, WIN_H - 40, WIN_W / 2 + bar_w, WIN_H - 30, fill="blue",
                                         tags="overlay")
            sens_text = f"SENSITIVITY: {eff_zoom:.2f}"
            if self.booster_level > 0: sens_text += " (AUTO)"
            self.canvas.create_text(WIN_W / 2, WIN_H - 20, text=sens_text, fill="black", font=("Arial", 10, "bold"),
                                    tags="overlay")

        else:
            self.canvas.create_text(WIN_W / 2, WIN_H / 2, text="DISCONNECTED", fill="red", font=("Arial", 40, "bold"),
                                    tags="overlay")

        self.root.after(5, self.update_gui)

    def draw_readout(self, x, y, label, value, color):
        self.canvas.create_rectangle(x - 90, y - 50, x + 90, y + 50, fill="#eeeeee", outline="black", tags="overlay")
        self.canvas.create_text(x, y - 20, text=label, fill="#555", font=("Arial", 10, "bold"), tags="overlay")
        self.canvas.create_text(x, y + 10, text=value, fill=color, font=("Arial", 30, "bold"), tags="overlay")

    def on_close(self):
        self.sensor.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProMeterAppV17(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()