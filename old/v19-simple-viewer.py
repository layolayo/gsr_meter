import tkinter as tk
from tkinter import ttk, filedialog
import csv
import bisect

# Minimalist Session Viewer (v19)
# Goal: Graph exact values in time order with Zoom & Time sliders.
# No complex audio sync, no smoothing, just raw data.

class SimpleViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("v19 Simple Viewer")
        self.root.geometry("1000x600")
        
        self.data_points = [] # List of (t, ta, setpoint)
        self.duration = 0.0
        
        # UI Setup
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Graph Canvas
        self.canvas = tk.Canvas(self.main_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Controls
        ctrl_frame = tk.Frame(root, height=100, bg="#eee")
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        btn_load = tk.Button(ctrl_frame, text="Load CSV", command=self.load_csv)
        btn_load.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Time Slider
        tk.Label(ctrl_frame, text="Time:").pack(side=tk.LEFT)
        self.scale_time = tk.Scale(ctrl_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=400, command=self.on_time_change)
        self.scale_time.pack(side=tk.LEFT, padx=10)
        
        # Zoom Slider (Window Size in seconds)
        tk.Label(ctrl_frame, text="Zoom (sec):").pack(side=tk.LEFT)
        self.scale_zoom = tk.Scale(ctrl_frame, from_=1, to=300, orient=tk.HORIZONTAL, length=200, command=self.on_draw)
        self.scale_zoom.set(30)
        self.scale_zoom.pack(side=tk.LEFT, padx=10)
        
        # Info Label
        self.lbl_info = tk.Label(ctrl_frame, text="Load a file to begin.", font=("Arial", 10))
        self.lbl_info.pack(side=tk.RIGHT, padx=10)

    def load_csv(self):
        fn = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fn: return
        
        raw_data = []
        try:
            with open(fn, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        t = float(row.get("Elapsed_Sec", 0))
                        ta = float(row.get("GSR_TA", 0))
                        sp = float(row.get("GSR_SetPoint", 0))
                        
                        # Basic validity check
                        if ta > 0.1:
                            raw_data.append({'t': t, 'ta': ta, 'sp': sp})
                    except: pass
            
            # [CRITICAL] SORT DATA because file has disorder
            raw_data.sort(key=lambda x: x['t'])
            
            # [CRITICAL] Normalize Time to 0.0
            if raw_data:
                t0 = raw_data[0]['t']
                for x in raw_data:
                    x['t'] -= t0
            
            self.data_points = raw_data
            if raw_data:
                self.duration = raw_data[-1]['t']
                self.scale_time.config(to=self.duration)
                self.scale_time.set(0)
                
                # Auto-scale Y (Global)
                all_ta = [x['ta'] for x in raw_data]
                self.y_min = min(all_ta)
                self.y_max = max(all_ta)
                self.lbl_info.config(text=f"Loaded {len(raw_data)} pts. Range: {self.y_min:.2f}-{self.y_max:.2f}")
                
            self.on_draw()
            
        except Exception as e:
            self.lbl_info.config(text=f"Error: {e}")

    def on_time_change(self, val):
        self.on_draw()
        
    def on_draw(self, _=None):
        self.canvas.delete("all")
        if not self.data_points: return
        
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 10 or H < 10: return
        
        t_center = self.scale_time.get()
        zoom = self.scale_zoom.get()
        
        # Window: [t_center - zoom/2, t_center + zoom/2]
        t_start = t_center - (zoom/2)
        t_end = t_center + (zoom/2)
        
        # Binary Search for subset
        keys = [x['t'] for x in self.data_points]
        idx_start = bisect.bisect_left(keys, t_start)
        idx_end = bisect.bisect_right(keys, t_end)
        
        subset = self.data_points[idx_start:idx_end]
        
        if not subset:
            self.canvas.create_text(W/2, H/2, text="(No Data in View)")
            return
            
        # Draw Logic
        # Y Mapping: Global Min/Max (Fixed)
        y_span = (self.y_max - self.y_min)
        if y_span == 0: y_span = 1
        
        points_ta = []
        points_sp = []
        
        # Decimation for performance if zoomed out
        # Max visual pixels ~1000
        stride = max(1, len(subset) // 1500)
        
        for i in range(0, len(subset), stride):
            pt = subset[i]
            x_norm = (pt['t'] - t_start) / zoom
            gx = x_norm * W
            
            # TA
            y_norm = (pt['ta'] - self.y_min) / y_span
            gy = H - (y_norm * H)
            points_ta.append(gx)
            points_ta.append(gy)
            
            # SetPoint
            y_norm_sp = (pt['sp'] - self.y_min) / y_span
            gy_sp = H - (y_norm_sp * H)
            points_sp.append(gx)
            points_sp.append(gy_sp)
            
        if len(points_ta) >= 4:
            self.canvas.create_line(points_ta, fill="green", width=2, tags="line")
        if len(points_sp) >= 4:
            self.canvas.create_line(points_sp, fill="blue", width=2, dash=(2,2), tags="line")
            
        # Center Line
        mid_x = 0.5 * W # Or exactly matches t_center? (t_center - t_start)/zoom = 0.5
        self.canvas.create_line(mid_x, 0, mid_x, H, fill="red")
        
        # Readout at Center
        # Find point closests to t_center
        idx_center = bisect.bisect_left(keys, t_center)
        if idx_center < len(self.data_points):
            val = self.data_points[idx_center]
            self.canvas.create_text(10, 10, anchor=tk.NW, text=f"Time: {val['t']:.2f}s\nTA: {val['ta']:.4f}\nSet: {val['sp']:.4f}", font=("Arial", 14, "bold"), fill="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleViewer(root)
    root.mainloop()
