
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import time
import math
import sys
import os
import csv
import numpy as np
import threading
import sounddevice as sd
import scipy.io.wavfile as wavfile
import random
from scipy.signal import savgol_filter # [NEW] For advanced smoothing (better than Moving Average)

# ==========================================
#           CONSTANTS (Reused from v16)
# ==========================================
WIN_W = 1450 # [FIX] Added 50px
WIN_H = 850

# [FIX] Centering Logic for Left Pane (approx 1150px wide)
# Center X = 575 (Still good)
PIVOT_X = 575
PIVOT_Y = 575
NEEDLE_LENGTH_MAX = 470
NEEDLE_LENGTH_MIN = 300
NEEDLE_WIDTH = 4
NEEDLE_COLOR = "#882222"
ANGLE_CENTER = 90
MAX_SWEEP = 40  # +/- 40 degrees
RESET_TARGET = 11 # [RESTORED] 11 degree offset as required

# [FIX] Maximize Graph Width
# [FIX] Maximize Graph Width (Shrink slightly to fit sliders)
GRAPH_W = 1020 
GRAPH_X = 80 # [FIX] Increased from 50 to 80 to fit Slider Labels
GRAPH_Y = 450 
GRAPH_H = 240 

SCALE_POS_X = 575
SCALE_POS_Y = 195

# ==========================================
#           DATA LOADER
# ==========================================
class SessionData:
    def __init__(self):
        self.csv_fn = ""
        self.wav_fn = ""
        self.csv_data = [] # Primary (GSR)
        self.audio_data = None
        self.audio_fs = 44100
        self.duration = 0.0
        self.loaded = False
        self.session_name = "" # [NEW] Stores "Session_YYYY-MM-DD..."

    def load_session(self, csv_path):
        self.csv_fn = csv_path
        self.csv_data = []
        
        # [NEW] Extract Session Name from Folder
        try:
            # path/to/Session_Data/Session_2026-01-03_12-00-00/gsr.csv
            folder = os.path.basename(os.path.dirname(csv_path))
            if "Session" in folder:
                 self.session_name = folder
            else:
                 # Fallback: Use filename if not in session folder
                 self.session_name = os.path.basename(csv_path)
        except:
            self.session_name = "Unknown Session"
        
        # [FIX] Clear Caches
        if hasattr(self, 'keys_cache'): del self.keys_cache
        
        # 1. Parse Primary CSV (GSR or Integrated)
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        item = {
                            "t": float(row.get("Elapsed_Sec", 0)),
                            "ta": float(row.get("GSR_TA", 0)),
                            "smoothed_ta": float(row.get("GSR_TA", 0)),
                            "center": float(row.get("GSR_SetPoint", 0)),
                            "sens": float(row.get("GSR_Sens", 0.5)),
                            # Legacy fallback (if loading old integrated usage)
                            "calm": float(row.get("EEG_Calm", 0)),
                            "focus": float(row.get("EEG_Focus", 0))
                        }
                        if item['ta'] > 0.1:
                            self.csv_data.append(item)
                    except: pass
                    
            if not self.csv_data: return False

            # Sort GSR
            self.csv_data.sort(key=lambda x: x['t'])
            
            # Normalize (Base Time)
            t_zero = self.csv_data[0]['t']
            if t_zero > 0.0:
                print(f"[DATA] Normalizing GSR by -{t_zero:.2f}s")
                for row in self.csv_data:
                    row['t'] -= t_zero

            # [NEW] Apply Smoothing to TA
            if len(self.csv_data) > 15:
                 raw_vals = [x['ta'] for x in self.csv_data]
                 try:
                     smooth_vals = savgol_filter(raw_vals, window_length=31, polyorder=3)
                     for i, x in enumerate(self.csv_data): x['smoothed_ta'] = smooth_vals[i]
                 except: pass

            self.duration = self.csv_data[-1]['t']
            
        except Exception as e:
            print(f"CSV Error: {e}")
            return False
        
        # 3. Find Audio (Heuristic)
        self.find_audio(csv_path, t_zero) # Pass t_zero if needed? No, audio is usually aligned to file start.
        
        # 4. Find Notes
        self.find_notes(csv_path)

        self.loaded = True
        return True

    def find_audio(self, csv_path, t_zero=0):
        # ... logic as before ...
        base = os.path.basename(csv_path)
        dir_name = os.path.dirname(csv_path)
        
        # Try exact name match (.csv -> .wav)
        candidates = [base.replace(".csv", ".wav")]
        
        # Heuristic: integrated_session_YYYY... -> integrated_audio_YYYY...
        # Heuristic: gsr.csv -> audio.wav (v21 convention)
        if base == "gsr.csv":
            candidates.append("audio.wav")
            
        # Search
        found = None
        for c in candidates:
            p = os.path.join(dir_name, c)
            if os.path.exists(p): found = p; break
            
        # Fallback: fuzzy timestamp search
        if not found:
             # Just look for ANY wav in the folder? No, risky. 
             # Use v18 logic: split timestamp.
             parts = base.split('_')
             if len(parts) >= 2:
                  # Try to find a wav containing the date part
                  pass 

        if found:
            print(f"Found Audio: {found}")
            try:
                # [MATCH V10] Use scipy.io.wavfile (Robust)
                fs, data = wavfile.read(found)
                
                # Convert to float32 if int
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                
                # Check silence
                if abs(data.max()) < 0.001:
                     print("WARNING: Audio seems silent!")
                
                self.audio_data = data
                self.audio_fs = fs
                self.wav_fn = found
                
                # Check Duration
                dur_aud = len(data) / fs
                print(f"[AUDIO] Loaded {dur_aud:.2f}s")
                self.duration = max(self.duration, dur_aud)
            except Exception as e: print(f"Audio Error: {e}")
        else:
            self.audio_data = None
            
    def find_notes(self, csv_path):
        # [FIX] Robustly find notes.txt regardless of which CSV was loaded
        folder = os.path.dirname(csv_path)
        self.notes_fn = os.path.join(folder, "notes.txt")
        
        if os.path.exists(self.notes_fn):
            try:
                with open(self.notes_fn, 'r') as f: self.notes_content = f.read()
            except Exception as e:
                print(f"Notes Read Error: {e}")
                self.notes_content = ""
        else: 
            self.notes_content = ""

    def get_state_at(self, t):
        import bisect
        
        # 1. GSR State (Primary)
        if not self.csv_data: return {'ta':0, 'center':0, 'sens':0}
        
        if not hasattr(self, 'keys_cache'): self.keys_cache = [x['t'] for x in self.csv_data]
        idx = bisect.bisect_left(self.keys_cache, t)
        if idx >= len(self.csv_data): idx = len(self.csv_data) - 1
        gsr_item = self.csv_data[idx]
        
        # [FIX] NO INTERPOLATION - Pure Data Playback
        # "Only map the exact data given - do not adjust it"
        return {
            "t": gsr_item.get("t"),
            "ta": gsr_item.get("ta"),
            "smoothed_ta": gsr_item.get("smoothed_ta"),
            "center": gsr_item.get("center", 0), 
            "sens": gsr_item.get("sens", 0.1)
        } 

# ==========================================
#           EVAL VIEW APP
# ==========================================
class SessionViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("GSR Session Viewer V17")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg="white")
        
        self.data = SessionData()
        self.is_playing = False
        self.start_time = 0
        self.current_t = 0.0
        self.play_offset = 0.0
        self.last_idx = 0
        
        self.play_offset = 0.0
        self.last_idx = 0
        
        self.audio_stream = None
        self.audio_idx = 0
        self.audio_devices = {}
        self.audio_device_index = "Default" # [MATCH V10] Start with Default
        # [MATCH V10] Do NOT query devices on init. It might tickle ALSA.
        # self.get_audio_devices()
        
        self._setup_ui()
        self._setup_menu()
        
        # [NEW] Config Persistence
        self.config_fn = "viewer_config.json"
        self.load_config()
        
        # [NEW] Handle Close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def load_config(self):
        import json
        try:
            if os.path.exists(self.config_fn):
                with open(self.config_fn, 'r') as f:
                    cfg = json.load(f)
                    
                # Apply Zoom (Time Window)
                if 'zoom_time' in cfg:
                    z = float(cfg['zoom_time'])
                    # Verify range
                    z = max(5, min(600, z))
                    self.scale_zoom.set(z)
                    print(f"Loaded Zoom: {z}s")
                    
                # Apply Init Dir if saved
                if 'last_dir' in cfg:
                     self.init_dir = cfg['last_dir']
                else:
                     self.init_dir = "Session_Data"
        except Exception as e:
            print(f"Config Load Error: {e}")
            self.init_dir = "Session_Data"
            
    def save_config(self):
        import json
        cfg = {}
        try:
            # Save Zoom
            cfg['zoom_time'] = self.scale_zoom.get()
            
            # Save Last Dir (from opened file)
            if self.data.csv_fn:
                 cfg['last_dir'] = os.path.dirname(self.data.csv_fn)
            elif hasattr(self, 'init_dir'):
                 cfg['last_dir'] = self.init_dir

            with open(self.config_fn, 'w') as f:
                json.dump(cfg, f)
            print("Config Saved.")
        except Exception as e:
            print(f"Config Save Error: {e}")

    def on_close(self):
        self.save_config()
        self.root.destroy()
        
    def _setup_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Session (CSV)...", command=self.open_file)
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

    def _setup_ui(self):
        # [NEW] Session Title Bar
        self.lbl_title = tk.Label(self.root, text="No Session Loaded", font=("Arial", 16, "bold"), bg="white", pady=10)
        self.lbl_title.pack(side=tk.TOP, fill=tk.X)
        
        # [NEW] Split Pane (Matches v10 Layout)
        self.panes = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5)
        self.panes.pack(fill=tk.BOTH, expand=True)

        # === LEFT PANE: Graph + Controls ===
        left_frame = tk.Frame(self.panes, bg="white")
        # [FIX] Relaxed slightly to 1150 (allows 300px for Notes)
        self.panes.add(left_frame, minsize=1150, stretch="always") # Main Graph Area

        # 1. Canvas (Inside Left Frame)
        # We pack canvas directly, no extra container needed if we use create_window
        self.canvas = tk.Canvas(left_frame, width=WIN_W, height=WIN_H-100, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # [NEW] Y-Axis Scale (VERTICAL SLIDER) - EXPONENTIAL MAPPING
        # Range 0-100 (Internal). Mapped to 0.0 - 6.5 visually.
        # to=0 because Tkinter Vertical Scale has 'from' at top usually.
        # But here 'from_=100' at top, 'to=0' at bottom works for level.
        self.scale_y_max = tk.Scale(self.canvas, from_=100, to=0, orient=tk.VERTICAL, bg="white", bd=1, showvalue=0)
        self.scale_y_max.set(80) # Default ~4.1
        
        # [NEW] Real-time updates via command
        self.scale_y_max.config(command=self.on_ymax_slide)
        
        # Bind release to final check
        self.scale_y_max.bind("<ButtonRelease-1>", self.on_zoom_change)
        
        # Place it
        # Position at (GRAPH_X - 5), centered vertically relative to Graph
        # anchor="e" means (GRAPH_X - 5) is the Right Edge of the slider.
        # [FIX] Reduced width from 50 to 30
        self.canvas.create_window(GRAPH_X - 5, GRAPH_Y + (GRAPH_H/2), window=self.scale_y_max, height=GRAPH_H, width=30, anchor="e")
        
        # [NEW] Value Label for Slider (Dynamic)
        self.canvas.create_text(GRAPH_X - 25, GRAPH_Y - 15, text="6.5", font=("Arial", 10, "bold"), fill="#555", tags="lbl_ymax")

        # -------------------------------------------------------------------------
        # [NEW] RIGHT Y-Axis Scale (MIN Value Control)
        # Position: GRAPH_X + GRAPH_W + 5 (Right side of graph)
        # -------------------------------------------------------------------------
        self.scale_y_min = tk.Scale(self.canvas, from_=100, to=0, orient=tk.VERTICAL, bg="white", bd=1, showvalue=0)
        self.scale_y_min.set(0) # Default 0 (Bottom of range)
        
        self.scale_y_min.config(command=self.on_ymin_slide)
        self.scale_y_min.bind("<ButtonRelease-1>", self.on_zoom_change)
        
        # Position at Right Edge
        slider_x = GRAPH_X + GRAPH_W + 5
        # [FIX] Reduced width from 50 to 30
        self.canvas.create_window(slider_x, GRAPH_Y + (GRAPH_H/2), window=self.scale_y_min, height=GRAPH_H, width=30, anchor="w")
        
        # Label for Min Slider (Bottom)
        self.canvas.create_text(slider_x + 15, GRAPH_Y + GRAPH_H + 15, text="0.0", font=("Arial", 10, "bold"), fill="#555", tags="lbl_ymin")
        
        # Load Dial Image
        try:
            self.img_dial = Image.open("dial_background.png") # Expect in CWD
            # [FIX] Do NOT resize, use original v16 dimensions (native)
            # self.img_dial = self.img_dial.resize((600, 450), Image.Resampling.LANCZOS)
            self.tk_dial = ImageTk.PhotoImage(self.img_dial)
            self.canvas.create_image(SCALE_POS_X, SCALE_POS_Y, image=self.tk_dial, anchor=tk.CENTER)
        except:
            self.canvas.create_oval(SCALE_POS_X-300, SCALE_POS_Y-225, SCALE_POS_X+300, SCALE_POS_Y+225, outline="#ccc")
            self.canvas.create_text(SCALE_POS_X, SCALE_POS_Y, text=f"MISSING: dial_background.png")

        # 2. Controls Frame (Inside Left Frame)
        ctrl_frame = tk.Frame(left_frame, height=100, bg="#eee")
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        btn_load = tk.Button(ctrl_frame, text="LOAD SESSION", command=self.open_file, bg="#ddd")
        btn_load.pack(side=tk.LEFT, padx=10)
        
        self.btn_play = tk.Button(ctrl_frame, text="PLAY", command=self.toggle_play, bg="#90ee90", width=10)
        self.btn_play.pack(side=tk.LEFT, padx=10)
        
        # Scrubber
        self.scrubber = ttk.Scale(ctrl_frame, from_=0, to=100, command=self.on_seek)
        self.scrubber.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
        # [NEW] Zoom Slider
        # Range: 5s (Fine) to 300s (Zoomed Out) or more
        tk.Label(ctrl_frame, text="Zoom:", bg="#eee").pack(side=tk.LEFT)
        self.scale_zoom = tk.Scale(ctrl_frame, from_=5, to=600, orient=tk.HORIZONTAL, length=100, showvalue=0)
        self.scale_zoom.set(10) # Default 10s
        self.scale_zoom.pack(side=tk.LEFT, padx=5)
        # Bind release to update graph (live update might be OK too, let's try live)
        self.scale_zoom.bind("<B1-Motion>", self.on_zoom_change)
        self.scale_zoom.bind("<ButtonRelease-1>", self.on_zoom_change)
        
        # Bind events for Seeking (prevent judder)
        self.scrubber.bind("<ButtonPress-1>", self.on_scrub_start)
        self.scrubber.bind("<ButtonRelease-1>", self.on_scrub_end)
        
        self.lbl_time = tk.Label(ctrl_frame, text="00:00 / 00:00", font=("Arial", 12), bg="#eee")
        self.lbl_time.pack(side=tk.LEFT, padx=10)
        
        self.lbl_status = tk.Label(left_frame, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # === RIGHT PANE: Notes ===
        # [FIX] Shrink Notes Pane (width=200, minsize=200)
        right_frame = tk.Frame(self.panes, width=200, bg="#fafafa")
        self.panes.add(right_frame, minsize=200, stretch="never")
        
        tk.Label(right_frame, text="Session Notes", font=('Arial', 12, 'bold'), bg="#fafafa").pack(pady=10)
        
        from tkinter import scrolledtext
        self.txt_notes = scrolledtext.ScrolledText(right_frame, height=20, wrap=tk.WORD, font=("Arial", 10))
        self.txt_notes.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.btn_save_notes = tk.Button(right_frame, text="Save Notes", command=self.save_notes, bg='#ddffdd', height=2)
        self.btn_save_notes.pack(fill=tk.X, padx=5, pady=10)
        
        # Stats Label
        tk.Label(right_frame, text="Statistics", font=('Arial', 12, 'bold'), bg="#fafafa").pack(pady=5)
        self.lbl_val_stats = tk.Label(right_frame, text="--", justify=tk.LEFT, bg="#fafafa")
        self.lbl_val_stats.pack(fill=tk.X, padx=5)

        self.scubbing = False

    # Removed get_audio_devices / cycle_audio_device / force_play_file as they are unused

    def open_file(self):
        fn = filedialog.askopenfilename(initialdir="Session_Data", filetypes=[("CSV Files", "*.csv")])
        if fn:
            if self.data.load_session(fn):
                # [FIX] Clear Viewer Cache (Force Rebalance/Rescale)
                if hasattr(self, 'full_data_cache'):
                    del self.full_data_cache
                    
                # [FIX] Use Smart Session Name (extraction logic in SessionData)
                display_name = self.data.session_name if self.data.session_name else os.path.basename(fn)
                
                self.root.title(f"Viewer - {display_name}")
                self.lbl_title.config(text=f"{display_name}")
                
                # Update status
                if hasattr(self, 'lbl_status'):
                     self.lbl_status.config(text=f"Loaded: {display_name} ({self.data.duration:.1f}s)")
                
                # [NEW] Configure Max TA Slider
                # Find global max from cache if possible, or loaded data
                vals = [x['smoothed_ta'] for x in self.data.csv_data]
                if vals:
                    # Filter zeros for finding a "reasonable" max if needed, but max is max.
                    g_max = max(vals)
                    
                    # User Request: Dynamic Range based on Session Min/Max
                    # "Use max and min of session as limits... never select actual min"
                    
                    # 1. Determine Dynamic Bounds
                    # Top: Session Max (g_max)
                    # Bottom: Session Min (min_ta) + Tiny Buffer (0.02)
                    
                    self.dyn_max = g_max
                    # Ensure min is valid
                    g_min = min(vals) if vals else 0
                    if g_min <= 0: g_min = 0.1
                    
                    self.dyn_min = g_min + 0.02 # [FIX] Reduced buffer significantly for Zoom
                    
                    # Safety: Ensure dyn_max > dyn_min
                    if self.dyn_max <= self.dyn_min: self.dyn_max = self.dyn_min + 0.1 # Relaxed safety
                    
                    # 2. Set Slider to Top (100 -> dyn_max)
                    self.scale_y_max.set(100)
                    self.current_manual_max = self.dyn_max
                    self.canvas.itemconfigure("lbl_ymax", text=f"{self.dyn_max:.2f}")
                    
                    # [NEW] 3. Set Min Slider to Bottom (0 -> dyn_min)
                    self.scale_y_min.set(0)
                    self.current_manual_min = self.dyn_min
                    self.canvas.itemconfigure("lbl_ymin", text=f"{self.dyn_min:.2f}")
                
                self.scrubber.config(to=self.data.duration)
                
                # [NEW] Populate Notes
                self.txt_notes.delete('1.0', tk.END)
                if self.data.notes_content:
                    self.txt_notes.insert(tk.END, self.data.notes_content)
                else:
                    self.txt_notes.insert(tk.END, "No notes found or created yet.\n")
                    
                # [NEW] Update Stats
                dur_m = int(self.data.duration // 60)
                dur_s = int(self.data.duration % 60)
                stats = f"Duration: {dur_m}m {dur_s}s\n"
                stats += f"\nCSV Rows: {len(self.data.csv_data)}"
                if self.data.audio_data is not None:
                     stats += f"\nAudio: {len(self.data.audio_data)/self.data.audio_fs:.1f}s"
                self.lbl_val_stats.config(text=stats)
                
                self.update_gui_once(0.0)
                self.current_t = 0.0
                self.last_idx = 0
            else:
                messagebox.showerror("Error", "Failed to load session data.")

    def save_notes(self):
        content = self.txt_notes.get("1.0", tk.END).strip()
        if not self.data.notes_fn:
            # Should have been set by load_session default, but safety check
            if self.data.csv_fn:
                self.data.notes_fn = self.data.csv_fn.replace(".csv", "_notes.txt")
            else:
                messagebox.showerror("Error", "No session loaded.")
                return

        try:
            with open(self.data.notes_fn, "w") as f:
                f.write(content)
            
            # Update cache
            self.data.notes_content = content
            messagebox.showinfo("Saved", f"Notes saved to:\n{os.path.basename(self.data.notes_fn)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save notes: {e}")

    def toggle_play(self):
        if self.is_playing:
            # Just pause (stop audio, keep offset)
            # Calculate where we are exact
            elapsed = time.time() - self.start_time
            self.current_t += elapsed
            self.pause(reset=False)
        else:
            self.play()

    def play(self):
        if self.is_playing: return
        if not self.data.loaded: return
        
        try:
            self.is_playing = True
            self.btn_play.config(text="PAUSE")

            # AUDIO
            if self.data.audio_data is not None:
                start_frame = int(self.current_t * self.data.audio_fs)
                if start_frame < len(self.data.audio_data):
                    # [MATCH V10 EXACTLY] Simple Slice Playback
                    sd.play(self.data.audio_data[start_frame:], self.data.audio_fs)
                    self.start_time = time.time()
                    print(f"[AUDIO] Playing from {self.current_t:.1f}s")
                    self._animate()
                else:
                    self.pause(reset=True)
            else:
                 # No audio, just animate
                 self.start_time = time.time()
                 self._animate()

        except Exception as e:
            print(f"Start Error: {e}")
            self.pause()

    def pause(self, reset=False):
        self.is_playing = False
        self.btn_play.config(text="PLAY")

        try:
            sd.stop()
        except:
            pass

        if reset:
            self.current_t = 0
            self.scubbing = True
            self.scrubber.set(0)
            self.scubbing = False
            self.update_gui_once(0)

    def on_seek(self, val):
         # Used for real-time label update only, NOT seeking audio
         t_val = float(val)
         self.lbl_time.config(text=f"{int(t_val//60):02}:{int(t_val%60):02} / {int(self.data.duration//60):02}:{int(self.data.duration%60):02}")
         
         # [NEW] Real-time Graph Update
         if self.scubbing:
             self.update_gui_once(t_val)

    def on_scrub_start(self, event):
        self.scubbing = True
        # [NEW] Seamless Scrubbing
        # If playing, pause temporarily so we don't fight the update loop
        if self.is_playing:
            self.was_playing_before_scrub = True
            # Stop audio but don't reset index
            self.pause(reset=False) 
        else:
            self.was_playing_before_scrub = False
        
    def on_scrub_end(self, event):
        self.scubbing = False
        val = self.scrubber.get()
        self.perform_seek(val)
        
        # [NEW] Resume if we were playing
        if getattr(self, 'was_playing_before_scrub', False):
            self.play()
            self.was_playing_before_scrub = False

    def perform_seek(self, val):
        # Update time state. Playback resumption is handled by on_scrub_end.
        # If user just clicks without drag (on_scrub_start might fire?), this logic still holds.
        self.current_t = float(val)
        self.update_gui_once(self.current_t)
        print(f"Seek to {self.current_t:.1f}s")
        
        # Explicitly update scrubber just in case
        self.scrubber.set(self.current_t)

    def _animate(self):
        if not self.is_playing: return

        # SYSTEM CLOCK DRIVE
        elapsed = time.time() - self.start_time
        current_time = self.current_t + elapsed

        if current_time >= self.data.duration:
            self.pause(reset=True)
            return

        # Only update slider if user is NOT dragging it
        if not self.scubbing:
            self.scrubber.set(current_time)

        self.update_gui_once(current_time)
        # [FIX] High Speed Refresh (15ms ~ 66fps)
        self.root.after(15, self._animate)

    def update_gui_once(self, t):
        # Update Scrubber
        if not self.scubbing:
            self.scrubber.set(t)
            
        # Time Label
        mins = int(t // 60)
        secs = int(t % 60)
        tot_m = int(self.data.duration // 60)
        tot_s = int(self.data.duration % 60)
        self.lbl_time.config(text=f"{mins:02}:{secs:02} / {tot_m:02}:{tot_s:02}")
        
        # Find State
        state = self.get_interpolated_state(t)
        if not state: return
        
        ta = state['ta']
        center = state['center']
        sens = state['sens']
        
        # Draw Needle
        self.draw_needle(ta, center, sens)
        
        # Draw Graph Update (Cursor)
        self.draw_graph(t)

        # Draw Readouts
        self.draw_readouts(state)

    def get_interpolated_state(self, t):
        # Linear scan from last_idx
        rows = self.data.csv_data
        if not rows: return None
        
        # Reset if seeking backwards
        if self.last_idx >= len(rows) or rows[self.last_idx]['t'] > t:
            self.last_idx = 0
            
        # Advance to find the interval [t0, t1] where t0 <= t <= t1
        # We want rows[i]['t'] <= t < rows[i+1]['t']
        while self.last_idx < len(rows) - 2 and rows[self.last_idx+1]['t'] < t:
            self.last_idx += 1
            
        # Now rows[self.last_idx] is the start point.
        row0 = rows[self.last_idx]
        
        # If at end
        if self.last_idx >= len(rows) - 1:
            return row0
            
        # [FIX] NO INTERPOLATION - Pure Data Playback
        # "Only map the exact data given - do not adjust it"
        return {
            "t": row0.get("t"),
            "ta": row0.get("ta"),
            "smoothed_ta": row0.get("smoothed_ta"),
            "center": row0.get("center", 0), 
            "sens": row0.get("sens", 0.1)
        }

    def draw_needle(self, ta, center, sensitivity=0.2):
        self.canvas.delete("needle_obj")
        
        # Calculate Angle
        
        # [FIX] Match Recorder Logic Exactly to Ensure Sync
        # Recorder: half_win = dial_range_ta / 2.0
        # Recorder: angle_v = (ratio * MAX_SWEEP) + RESET_TARGET
        # Reset Trigger: abs(angle_v) > MAX_SWEEP (i.e. > 40 or < -40)
        
        diff = ta - center
        
        zoom = sensitivity if sensitivity > 0 else 0.2
        half_win = zoom / 2.0
        
        ratio = diff / half_win
        
        # Clamp visual (Optional, to keep inside dial if data goes slightly out)
        # But mostly we want to see it go to the limit.
        ratio = max(-2.0, min(2.0, ratio)) # Loose clamp
        
        # Standard Calculation (No asymmetric scaling hacks)
        angle_v = (ratio * MAX_SWEEP) + RESET_TARGET
        
        # Visual Clamp to Dial Limits (+/- 40)
        angle_v = max(-MAX_SWEEP, min(MAX_SWEEP, angle_v))
        
        rad = math.radians(ANGLE_CENTER + angle_v)
        
        tx = PIVOT_X + NEEDLE_LENGTH_MAX * math.cos(rad)
        ty = PIVOT_Y - NEEDLE_LENGTH_MAX * math.sin(rad)
        bx = PIVOT_X + NEEDLE_LENGTH_MIN * math.cos(rad)
        by = PIVOT_Y - NEEDLE_LENGTH_MIN * math.sin(rad)
        
        self.canvas.create_line(bx, by, tx, ty, width=NEEDLE_WIDTH, fill=NEEDLE_COLOR, capstyle=tk.ROUND, tags="needle_obj")

    def on_ymax_slide(self, val):
        # [NEW] Dynamic Mapping based on Session Limits
        # Range: [self.dyn_min, self.dyn_max]
        # Slider val: 0-100
        
        # Check if dynamic limits exist (might slide before file loaded?)
        d_min = getattr(self, 'dyn_min', 0.1)
        d_max = getattr(self, 'dyn_max', 6.5)
        
        try:
            x = float(val)
        except: 
            return
            
        # Linear Interpolation: v = min + (max - min) * (x / 100)
        mapped_val = d_min + (d_max - d_min) * (x / 100.0)
        
        # [NEW] Clamp against Current Min (Prevent Inversion)
        cur_min = getattr(self, 'current_manual_min', self.min_ta)
        if mapped_val <= cur_min + 0.01:
            mapped_val = cur_min + 0.01
            # Optional: Snap slider back? 
            # safe_x = (mapped_val - d_min) / (d_max - d_min) * 100
            # self.scale_y_max.set(safe_x) # Might cause recursion loop?
        
        self.current_manual_max = mapped_val
        
        # Update Label
        self.canvas.itemconfigure("lbl_ymax", text=f"{mapped_val:.2f}")
        
        # Redraw
        self.update_gui_once(self.current_t)

    def on_zoom_change(self, event):
        # Update without seek
        self.update_gui_once(self.current_t)

    def draw_graph(self, t):
        self.canvas.delete("graph_cursor")
        self.canvas.delete("graph_bg")
        self.canvas.delete("graph_line")
        
        # Background
        self.canvas.create_rectangle(GRAPH_X, GRAPH_Y, GRAPH_X+GRAPH_W, GRAPH_Y+GRAPH_H, fill="#eee", tags="graph_bg")
        
        # [NEW] SCROLLING WINDOW logic via Slider
        try:
            WINDOW_SIZE = float(self.scale_zoom.get())
        except:
            WINDOW_SIZE = 10.0
        
        # [FIX] Offset Cursor Logic
        # User wants cursor at 30% mark to see more future context.
        # History: 30%, Future: 70%
        history_dur = WINDOW_SIZE * 0.3
        future_dur  = WINDOW_SIZE * 0.7
        
        t_start = t - history_dur
        t_end = t + future_dur
        
        if not self.data.csv_data: return
             
        # Extract points in range
        if not hasattr(self, 'full_data_cache') and self.data.loaded:
            # [FIX] Apply Time Normalization Check (if not done in load)
            # Just to be safe, though load_session should handle it.
            pass
            
            # [FIX] Use SMOOTHED data for Graph (prevents jagged lines)
            self.full_data_cache = [(x['t'], x['smoothed_ta']) for x in self.data.csv_data]
            vals = [x[1] for x in self.full_data_cache]
            
            # [FIX] Robust Scaling: Filter out 0.0 or effectively zero values
            # This prevents the graph from being squashed if there are startup glitches
            if vals:
                 valid_vals = [v for v in vals if v > 0.1] # Filter drops/glitches
                 if not valid_vals: valid_vals = vals # Fallback if all 0
                 self.min_ta = min(valid_vals)
                 self.max_ta = max(vals) # Still use true max for top
            else:
                 self.min_ta = 0
                 self.max_ta = 1
                 
            self.span_ta = self.max_ta - self.min_ta if self.max_ta != self.min_ta else 1.0
            self.keys_cache = [x[0] for x in self.full_data_cache]
            
        if not hasattr(self, 'full_data_cache'): return
        
        # [NEW] Check Manual Max Override (For Clipping Artifacts)
        # Use cached mapped value if available
        manual_max = getattr(self, 'current_manual_max', self.max_ta)
        # [NEW] Check Manual Min Override
        manual_min = getattr(self, 'current_manual_min', self.min_ta)
             
        # Drawing Logic
        # Use manual_max as the "Visual Top"
        eff_max = manual_max
        eff_min = manual_min # Use Manual Min
        
        # Ensure span > 0
        if eff_max <= eff_min: eff_max = eff_min + 1.0
        eff_span = eff_max - eff_min

        # [NEW] Vertical Padding Logic (5% top, 5% bottom based on EFFECTIVE span)
        # Ensure span > tiny amount (avoid div/0)
        # Relaxed logic: allow tiny spans for high zoom
        if eff_max <= eff_min + 0.001: eff_max = eff_min + 0.001
        
        eff_span = eff_max - eff_min
        
        padding = eff_span * 0.05
        v_min = eff_min - padding
        v_span = eff_span + (padding * 2)



        # [FIX] Restore Bisect Logic to define subset_data
        import bisect
        start_idx = bisect.bisect_left(self.keys_cache, t_start)
        end_idx = bisect.bisect_right(self.keys_cache, t_end)
        
        start_idx = max(0, start_idx)
        end_idx = min(len(self.data.csv_data), end_idx)
        
        if start_idx >= end_idx: return
        
        subset_data = self.full_data_cache[start_idx:end_idx] 
        if not subset_data: return

        time_span = t_end - t_start 
        if time_span <= 0: return

        points = []
        count = len(subset_data)
        
        # [FIX] revert to Stride (Simple Decimation)
        # GSR data is slow-moving/smooth. Min-Max envelope creates visual artifacts ("missing parts").
        # Decimation is safe for this signal type.
        
        TARGET_POINTS = 3000 # Increased density slightly
        stride = max(1, count // TARGET_POINTS)

        for i in range(0, count, stride):
            pt = subset_data[i]
            x_time = pt[0]
            y_val = pt[1]
            
            # Map X
            x_rel = x_time - t_start
            x_norm = x_rel / WINDOW_SIZE
            gx = GRAPH_X + (x_norm * GRAPH_W)
            
            # Map Y
            # Map Y
            # [FIX] Clamp data to Visual Limits (eff_min/eff_max) BEFORE mapping
            # This ensures that "off scale" data plots at the Max/Min lines (95%/5%)
            # and does NOT touch the graph border (0%/100%), respecting the padding.
            y_clamped = max(eff_min, min(eff_max, y_val))
            
            y_norm = (y_clamped - v_min) / v_span
            y_norm = max(0.0, min(1.0, y_norm))
            gy = GRAPH_Y + GRAPH_H - (y_norm * GRAPH_H)
            
            points.append(gx)
            points.append(gy)
            
        if len(points) >= 4:
            # [FIX] Revert Color to Green
            self.canvas.create_line(points, fill="#22dd22", width=2, tags="graph_line")
            
        # Draw Cursor (30% Mark)
        # cx = GRAPH_X + (GRAPH_W * 0.3)
        cx = GRAPH_X + (GRAPH_W * 0.3)
        self.canvas.create_line(cx, GRAPH_Y, cx, GRAPH_Y+GRAPH_H, fill="blue", dash=(2,2), tags="graph_cursor")
        
        
        # Overlay Time Labels on Graph Edges
        self.canvas.create_text(GRAPH_X + 5, GRAPH_Y + 10, text=f"{t_start:.1f}s", anchor=tk.NW, font=("Arial", 8), fill="#999", tags="graph_bg")
        self.canvas.create_text(GRAPH_X + GRAPH_W - 5, GRAPH_Y + 10, text=f"{t_end:.1f}s", anchor=tk.NE, font=("Arial", 8), fill="#999", tags="graph_bg")
        
        # Verify Audio Loading Status (Debug UI)
        wav_stat = "Loaded" if self.data.audio_data is not None else "None"
        if hasattr(self, 'lbl_status'):
             self.lbl_status.config(text=f"Time: {t:.1f}s | Audio: {wav_stat} | Window: {t_start:.1f}-{t_end:.1f}")

    def draw_readouts(self, state):
        self.canvas.delete("readout")
        
        # Move up below graph (Graph Y=450, H=240 -> Bot=690)
        y_lbl = 700
        y_val = 720
        
        # [FIX] Centered Readouts
        cx = GRAPH_X + (GRAPH_W / 2)
        spacing = 250
        
        x_ta = cx - spacing
        x_center = cx
        x_sens = cx + spacing
        
        # TA
        self.canvas.create_text(x_ta, y_lbl, text="INST TA", font=("Arial", 10), fill="#777", tags="readout")
        self.canvas.create_text(x_ta, y_val, text=f"{state['ta']:.4f}", font=("Arial", 20, "bold"), fill="black", tags="readout")
        
        # Center
        self.canvas.create_text(x_center, y_lbl, text="TA SET", font=("Arial", 10), fill="#777", tags="readout")
        self.canvas.create_text(x_center, y_val, text=f"{state['center']:.4f}", font=("Arial", 20, "bold"), fill="black", tags="readout")
        
        # Sens
        self.canvas.create_text(x_sens, y_val, text=f"{state['sens']:.4f}", font=("Arial", 20, "bold"), fill="black", tags="readout")

    def on_ymin_slide(self, val):
        # [NEW] Handler for Min Slider
        # Range: [self.dyn_min, self.dyn_max]
        d_min = getattr(self, 'dyn_min', 0.1)
        d_max = getattr(self, 'dyn_max', 6.5)
        
        try:
            x = float(val)
        except: 
            return
            
        # Linear Interpolation
        mapped_val = d_min + (d_max - d_min) * (x / 100.0)
        
        # [NEW] Clamp against Current Max (Prevent Inversion)
        cur_max = getattr(self, 'current_manual_max', self.max_ta)
        if mapped_val >= cur_max - 0.01:
            mapped_val = cur_max - 0.01
        
        self.current_manual_min = mapped_val
        
        # Update Label
        self.canvas.itemconfigure("lbl_ymin", text=f"{mapped_val:.2f}")
        
        # Redraw
        self.update_gui_once(self.current_t)


if __name__ == "__main__":
       
    root = tk.Tk()
    app = SessionViewer(root)
    root.mainloop()

