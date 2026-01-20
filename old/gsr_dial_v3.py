import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import hid
import threading
import time
import math
import sys
import os

# --- FILES ---
BG_FILE = "Background.jpg"
SCALE_FILE = "dial_background.png"

# --- CONFIGURATION ---
VENDOR_ID = 0x1fc9
PRODUCT_ID = 0x0003
V_SOURCE = 6.371
R_REF = 83.4

# --- WINDOW SIZE (Forcing standard size) ---
WIN_W = 1024
WIN_H = 768

# --- CALIBRATION (Relative to 1024x768) ---
PIVOT_X = 512  # Center width
PIVOT_Y = 575  # Lower middle
SCALE_POS_X = 512
SCALE_POS_Y = 195

# --- NEEDLE GEOMETRY ---
NEEDLE_COLOR = "#222222"
NEEDLE_LENGTH = 610
NEEDLE_WIDTH_BASE = 12
NEEDLE_WIDTH_TIP = 3

# --- DIAL SETTINGS ---
ANGLE_CENTER = 90
MAX_SWEEP = 40
RESET_TARGET = 11


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
            h.set_nonblocking(1)
            self.connected = True
            while self.running:
                try:
                    data = h.read(64)
                except:
                    self.connected = False;
                    time.sleep(1);
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
                time.sleep(0.01)
        except:
            self.connected = False

    def stop(self):
        self.running = False


class ProMeterAppV5:
    def __init__(self, root):
        self.root = root
        self.root.title("GSR Professional Meter V5")

        self.center_ta = 2.0
        self.zoom_window = 0.20
        self.p_x = PIVOT_X;
        self.p_y = PIVOT_Y
        self.s_x = SCALE_POS_X;
        self.s_y = SCALE_POS_Y

        # 1. Load & Resize Background
        try:
            raw_bg = Image.open(BG_FILE).convert("RGBA")
            # Force resize to standard window
            raw_bg = raw_bg.resize((WIN_W, WIN_H), Image.LANCZOS)
            self.bg_img = ImageTk.PhotoImage(raw_bg)
            print(f"Background resized to {WIN_W}x{WIN_H}")
        except:
            print("Background not found, using grey.")
            self.bg_img = None

        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.canvas = tk.Canvas(root, width=WIN_W, height=WIN_H, highlightthickness=0)
        self.canvas.pack()

        if self.bg_img:
            self.canvas.create_image(0, 0, image=self.bg_img, anchor=tk.NW)
        else:
            self.canvas.configure(bg="#888888")

        # 2. Scale (Standardized)
        try:
            self.scale_pil = Image.open(SCALE_FILE).convert("RGBA")
            # Optional: Resize scale if it's too huge
            # self.scale_pil = self.scale_pil.resize((600, 200), Image.LANCZOS)
            self.scale_img = ImageTk.PhotoImage(self.scale_pil)
            self.scale_id = self.canvas.create_image(self.s_x, self.s_y, image=self.scale_img, anchor=tk.CENTER)
        except:
            self.scale_img = None
            self.scale_id = None

        # --- CONTROLS ---
        root.bind("<Key>", self.handle_keypress)
        root.bind("<space>", lambda e: self.recenter_needle())

        self.sensor = GSRReader()
        self.sensor.start()
        self.update_gui()

    def handle_keypress(self, event):
        k = event.keysym.lower()
        c = event.char

        if c in ['=', '+'] or 'plus' in k or 'equal' in k:
            self.adj_zoom(-0.05)
        elif c in ['-', '_'] or 'minus' in k or 'under' in k:
            self.adj_zoom(0.05)

    def mv_p(self, x, y):
        self.p_x += x; self.p_y += y; print(f"PIVOT: {self.p_x},{self.p_y}")

    def mv_s(self, x, y):
        self.s_x += x; self.s_y += y; print(f"SCALE: {self.s_x},{self.s_y}")

    def adj_zoom(self, amount):
        self.zoom_window = max(0.05, self.zoom_window + amount)
        print(f"SENSITIVITY: {self.zoom_window:.2f}")

    def recenter_needle(self):
        target_ratio = RESET_TARGET / MAX_SWEEP
        half_win = self.zoom_window / 2.0
        if self.sensor.connected:
            self.center_ta = self.sensor.current_ta - (target_ratio * half_win)

    def update_gui(self):
        ta = self.sensor.current_ta
        is_conn = self.sensor.connected

        # 1. Update Scale
        if self.scale_id: self.canvas.coords(self.scale_id, self.s_x, self.s_y)

        # 2. Clear Dynamic Layers
        self.canvas.delete("needle_obj")
        self.canvas.delete("overlay")  # Clears text and graphics

        if is_conn:
            # Needle Logic
            half_win = self.zoom_window / 2.0
            diff = ta - self.center_ta
            ratio = diff / half_win
            angle_center = ratio * MAX_SWEEP

            if abs(angle_center) > MAX_SWEEP:
                self.recenter_needle()
                diff = ta - self.center_ta
                ratio = diff / half_win
                angle_center = ratio * MAX_SWEEP

            angle_rad = math.radians(ANGLE_CENTER + angle_center)

            # --- DRAW NEEDLE ---
            tip_x = self.p_x + NEEDLE_LENGTH * math.cos(angle_rad)
            tip_y = self.p_y - NEEDLE_LENGTH * math.sin(angle_rad)

            bx_r = self.p_x + (NEEDLE_WIDTH_BASE / 2) * math.cos(angle_rad - math.pi / 2)
            by_r = self.p_y - (NEEDLE_WIDTH_BASE / 2) * math.sin(angle_rad - math.pi / 2)
            bx_l = self.p_x + (NEEDLE_WIDTH_BASE / 2) * math.cos(angle_rad + math.pi / 2)
            by_l = self.p_y - (NEEDLE_WIDTH_BASE / 2) * math.sin(angle_rad + math.pi / 2)

            self.canvas.create_polygon(tip_x, tip_y, bx_r, by_r, bx_l, by_l,
                                       fill=NEEDLE_COLOR, outline=NEEDLE_COLOR, tags="needle_obj", smooth=True)
            r = NEEDLE_WIDTH_BASE / 2
            self.canvas.create_oval(self.p_x - r, self.p_y - r, self.p_x + r, self.p_y + r, fill=NEEDLE_COLOR,
                                    tags="needle_obj")

            # --- DRAW TEXT BOXES (Guaranteed Visibility) ---
            # Box 1: TA Reading (Left)
            self.draw_readout(200, WIN_H - 100, "TA", f"{self.center_ta:.2f}", "#000000")

            # Box 2: INST Reading (Right)
            self.draw_readout(WIN_W - 200, WIN_H - 100, "INST", f"{ta:.3f}", "#000000")

            # Box 3: Sensitivity Bar (Bottom Center)
            bar_w = int((1.0 / self.zoom_window) * 25)  # Wider bar = More sensitive
            self.canvas.create_rectangle(WIN_W / 2 - bar_w, WIN_H - 40, WIN_W / 2 + bar_w, WIN_H - 30, fill="blue",
                                         tags="overlay")
            self.canvas.create_text(WIN_W / 2, WIN_H - 20, text=f"SENSITIVITY: {self.zoom_window:.2f}", fill="black",
                                    font=("Arial", 10, "bold"), tags="overlay")

        else:
            # Disconnected
            self.canvas.create_text(WIN_W / 2, WIN_H / 2, text="DISCONNECTED", fill="red", font=("Arial", 40, "bold"),
                                    tags="overlay")
            self.draw_readout(200, WIN_H - 100, "TA", "---", "#555")
            self.draw_readout(WIN_W - 200, WIN_H - 100, "INST", "---", "#555")

        self.root.after(20, self.update_gui)

    def draw_readout(self, x, y, label, value, color):
        # Draw a semi-transparent white box behind text to ensure readability
        # Box coords: x-80, y-40 to x+80, y+40
        self.canvas.create_rectangle(x - 90, y - 50, x + 90, y + 50, fill="#eeeeee", outline="black", tags="overlay")

        self.canvas.create_text(x, y - 20, text=label, fill="#555", font=("Arial", 12, "bold"), tags="overlay")
        self.canvas.create_text(x, y + 10, text=value, fill=color, font=("Arial", 30, "bold"), tags="overlay")

    def on_close(self):
        self.sensor.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProMeterAppV5(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()