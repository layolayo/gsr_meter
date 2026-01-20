
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
import soundfile as sf
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
RESET_TARGET = 11

# [FIX] Maximize Graph Width
GRAPH_W = 1050 
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
        self.csv_data = [] # List of dicts
        self.audio_data = None
        self.audio_fs = 44100
        self.duration = 0.0
        self.loaded = False

    def load_session(self, csv_path):
        self.csv_fn = csv_path
        self.csv_data = []
        
        # [FIX] Clear Cache on Load
        if hasattr(self, 'keys_cache'):
            del self.keys_cache
        
        # 1. Parse CSV
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse relevant columns
                    try:
                        item = {
                            "t": float(row.get("Elapsed_Sec", 0)),
                            "ta": float(row.get("GSR_TA", 0)),
                            "smoothed_ta": float(row.get("GSR_TA", 0)), # Init with raw
                            "calm": float(row.get("EEG_Calm", 0)),
                            "focus": float(row.get("EEG_Focus", 0)),
                            "center": float(row.get("GSR_SetPoint", 0)),
                            "sens": float(row.get("GSR_Sens", 0.5))
                        }
                        # [FIX] Filter out Zeros (Startup artifacts)
                        if item['ta'] > 0.1:
                            self.csv_data.append(item)
                    except: pass
                    
            # [NEW] Normalize Timestamps (Start at 0.0)
            if self.csv_data:
                t_zero = self.csv_data[0]['t']
                if t_zero > 0.0:
                    print(f"[DATA] Normalizing timestamps by -{t_zero:.2f}s")
                    for row in self.csv_data:
                        row['t'] -= t_zero
                        
            # [FIX] Force Sort by Time
            # Dataset has disorder (timestamps jumping back), which breaks bisect.
            if self.csv_data:
                self.csv_data.sort(key=lambda x: x['t'])
                        
            # [NEW] Apply Smoothing
            if len(self.csv_data) > 15:
                 from scipy.signal import savgol_filter
                 raw_vals = [x['ta'] for x in self.csv_data]
                 try:
                     smooth_vals = savgol_filter(raw_vals, window_length=31, polyorder=3)
                     for i, x in enumerate(self.csv_data):
                         x['smoothed_ta'] = smooth_vals[i]
                 except: pass

        except Exception as e:
            print(f"CSV Error: {e}")
            return False

        if not self.csv_data: return False
        
        # [NEW] Normalization Logic MOVED TO SessionData.load_file
        # Ensures all internal state (timestamps, smoothing) is consistent.
        
        # [NEW] Apply Savitzky-Golay Filter (User requested "Finer Distinction")
        # Moving Average blurs peaks. SavGol preserves features (peaks/valleys) while smoothing noise/steps.
        if len(self.csv_data) > 15:
             raw_vals = [x['ta'] for x in self.csv_data]
             try:
                 # Window=31 (approx 0.25s), Polyorder=3 (Cubic)
                 # This smooths the steps into curves but keeps the magnitude of spikes.
                 smooth_vals = savgol_filter(raw_vals, window_length=31, polyorder=3)
                 
                 for i, x in enumerate(self.csv_data):
                     x['smoothed_ta'] = smooth_vals[i]
             except Exception as e:
                 print(f"Smoothing Error: {e}")

        if not self.csv_data: return False
        self.duration = self.csv_data[-1]['t']
        
        # 2. Find Audio
        # Format: Session_Data/Integrated_Session_YYYY-MM-DD_HH-MM-SS.csv
        # Audio: Session_Data/GSR_Audio_YYYY-MM-DD_HH-MM-SS.wav
        # Try heuristics based on timestamp part
        
        base = os.path.basename(csv_path)
        dir_name = os.path.dirname(csv_path)
        
        # Extract timestamp... assume standard format
        # Integrated_Session_2026-01-02_20-33-49.csv -> 2026-01-02_20-33-49
        ts_part = ""
        parts = base.split('_')
        if len(parts) >= 3:
            # Join the last two parts (Date + Time.csv)
            # Remove extension from last part
            p_last = parts[-1].replace(".csv", "")
            p_date = parts[-2]
            ts_part = f"{p_date}_{p_last}"
        
        # Search for wav
        # Strategy:
        # 1. Exact name match (.csv -> .wav)
        # 2. Relaxed timestamp match (ignoring prefix)
        
        wav_candidates = [
            base.replace(".csv", ".wav"),
            f"GSR_Audio_{ts_part}.wav",
            f"Integrated_Audio_{ts_part}.wav"
        ]
        
        # 3. Fuzzy Search in directory
        # Find any WAV that contains the timestamp string
        if ts_part and len(ts_part) > 10:
             for f in os.listdir(dir_name):
                 if f.endswith(".wav") and ts_part in f:
                     wav_candidates.append(f)
        
        found_wav = None
        for c in wav_candidates:
            p = os.path.join(dir_name, c)
            if os.path.exists(p):
                found_wav = p
                break
        
        if found_wav:
            print(f"Found Audio: {found_wav}")
            try:
                # [MATCH V10] Use float32 directly in read
                data, fs = sf.read(found_wav, dtype='float32')
                
                # Check silence
                if abs(data.max()) < 0.001:
                     print("WARNING: Audio seems silent!")
                
                # [MATCH V10] NO UPMIXING. Let the driver handle Mono.
                # v10 didn't upmix, and it worked.
                
                # [MATCH V10] Load Float32
                self.audio_data = data
                self.audio_fs = fs
                self.wav_fn = found_wav
                
                # Check Duration
                dur_aud = len(data) / fs
                print(f"[AUDIO] Loaded {dur_aud:.2f}s, {fs}Hz")
                if dur_aud > self.duration: 
                     if self.duration == 0: self.duration = dur_aud
                     self.duration = max(self.duration, dur_aud)
                     
            except Exception as e:
                 print(f"Audio Load Error: {e}")
                 self.audio_data = None
        else:
            print("No matching audio found.")
            # Silent placeholder if no audio found
            self.audio_data = np.zeros((44100*1, 2), dtype=np.float32)
            self.audio_fs = 44100

        # 3. Find Notes [NEW]
        # v16 pattern: filename.csv -> filename_notes.txt
        self.notes_fn = None
        self.notes_content = ""
        
        notes_candidates = [
            csv_path.replace(".csv", "_notes.txt"),
            csv_path.replace(".csv", ".txt"),
            os.path.join(dir_name, f"GSR_Audio_{ts_part}_notes.txt")
        ]
        
        for c in notes_candidates:
            if os.path.exists(c):
                self.notes_fn = c
                try:
                    with open(c, 'r') as f:
                        self.notes_content = f.read()
                    print(f"[NOTES] Loaded: {c}")
                    break
                except: pass
        
        if not self.notes_fn:
            # Set default path for saving if user creates notes
            self.notes_fn = csv_path.replace(".csv", "_notes.txt")
            print("[NOTES] No existing notes found.")

        self.loaded = True
        return True

    def get_state_at(self, t):
        # Binary search
        import bisect
        
        if not self.csv_data:
            return {'ta':0, 'center':0, 'sens':0}
            
        # Optimization: Keys cache
        if not hasattr(self, 'keys_cache'):
            self.keys_cache = [x['t'] for x in self.csv_data]
            
        # Find index
        idx = bisect.bisect_left(self.keys_cache, t)
        
        # Clamp Logic
        if idx >= len(self.csv_data): idx = len(self.csv_data) - 1
        if idx < 0: idx = 0
            
        item = self.csv_data[idx]
        
        # [DEBUG] Check if we are "drifting"
        # If t is way beyond item['t'], maybe return item but warn?
        # But here we just return the nearest state.
        
        return {
            'ta': item.get('smoothed_ta', item['ta']),
            'center': item['center'],
            'sens': item['sens']
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
        self.scale_y_max = tk.Scale(self.canvas, from_=100, to=0, orient=tk.VERTICAL, label="Max", bg="white", bd=1, showvalue=0)
        self.scale_y_max.set(80) # Default ~4.1
        
        # [NEW] Real-time updates via command
        self.scale_y_max.config(command=self.on_ymax_slide)
        
        # Bind release to final check
        self.scale_y_max.bind("<ButtonRelease-1>", self.on_zoom_change)
        
        # Place it
        # Position at (GRAPH_X - 5), centered vertically relative to Graph
        # anchor="e" means (GRAPH_X - 5) is the Right Edge of the slider.
        self.canvas.create_window(GRAPH_X - 5, GRAPH_Y + (GRAPH_H/2), window=self.scale_y_max, height=GRAPH_H, width=50, anchor="e")
        
        # [NEW] Value Label for Slider (Dynamic)
        self.canvas.create_text(GRAPH_X - 25, GRAPH_Y - 15, text="6.5", font=("Arial", 10, "bold"), fill="#555", tags="lbl_ymax")
        
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
                    
                fname = os.path.basename(fn)
                self.root.title(f"Viewer - {fname}")
                
                # [NEW] Update Title Label
                # Try to parse date/time for a nicer title
                try:
                    # Format: Integrated_Session_YYYY-MM-DD_HH-MM-SS.csv
                    parts = fname.replace(".csv", "").split('_')
                    if len(parts) >= 3:
                        dt = f"{parts[-2]} {parts[-1].replace('-', ':')}"
                        self.lbl_title.config(text=f"Session: {dt}")
                    else:
                        self.lbl_title.config(text=f"Session: {fname}")
                except:
                    self.lbl_title.config(text=f"Session: {fname}")
                
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
        self.root.after(20, self._animate)

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
            
        row1 = rows[self.last_idx + 1]
        
        t0 = row0['t']
        t1 = row1['t']
        
        # Avoid divide by zero
        if t1 <= t0: return row0
        
        # Interpolation Factor
        alpha = (t - t0) / (t1 - t0)
        # Clamp alpha (shouldn't happen with logic above but safety)
        alpha = max(0.0, min(1.0, alpha))
        
        # Helper lerp
        def lerp(k):
            v0 = row0.get(k, 0)
            v1 = row1.get(k, 0)
            return v0 + (v1 - v0) * alpha
            
        return {
            "t": t,
            "ta": lerp("smoothed_ta"), # [FIX] Return Smoothed TA for graph/needle
            "raw_ta": lerp("ta"),      # Keep Raw for reference if needed
            "center": lerp("center"),
            "sens": lerp("sens")
        }

    def draw_needle(self, ta, center, sensitivity=0.2):
        self.canvas.delete("needle_obj")
        
        # Calculate Angle
        # Basic Logic from v16:
        # diff = ta - center
        # ratio = diff / sensitivity
        # angle = ratio * MAX_SWEEP
        
        # Wait, v16 logic:
        # half_win = sensitivity # Actually base_sensitivity * zoom logic
        # simple approx:
        
        diff = ta - center
        # Sensitivity in CSV is 'GSR_Sens' which is 'eff_zoom'
        # [FIX] V16 Logic uses half-window for ratio calc
        zoom = sensitivity if sensitivity > 0 else 0.2
        half_win = zoom / 2.0
        
        ratio = diff / half_win
        
        # Clamp visual
        ratio = max(-1.2, min(1.2, ratio))
        
        angle_v = (ratio * MAX_SWEEP) + RESET_TARGET
        
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
             
        # Drawing Logic
        # Use manual_max as the "Visual Top"
        eff_max = manual_max
        eff_min = self.min_ta
        
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
            y_norm = (y_val - v_min) / v_span
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
        self.canvas.create_text(x_sens, y_lbl, text="SENS", font=("Arial", 10), fill="#777", tags="readout")
        self.canvas.create_text(x_sens, y_val, text=f"{state['sens']:.4f}", font=("Arial", 20, "bold"), fill="black", tags="readout")


if __name__ == "__main__":
       
    root = tk.Tk()
    app = SessionViewer(root)
    root.mainloop()

