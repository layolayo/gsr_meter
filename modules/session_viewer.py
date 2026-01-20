import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
# [MOD] Removed direct sounddevice usage for playback logic (delegated)
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, Checkbutton, BooleanVar
import os
import time
import sys
from datetime import datetime

# --- SETTINGS ---
WINDOW_SEC = 10.0  # [MOD] Increased to 10s to match main app
UPDATE_INTERVAL_MS = 50  # 20 FPS
DEFAULT_SMOOTHING = 6
DEFAULT_Y_SCALE = 100

class SessionViewer:
    def __init__(self, master_frame, audio_handler, on_close_callback=None):
        """
        master_frame: The parent tk.Widget (usually a Frame) to pack into.
        audio_handler: Instance of modules.audio_handler.AudioHandler
        on_close_callback: Function to call when user clicks 'Close' or 'Back'.
        """
        self.master = master_frame
        self.audio_handler = audio_handler # [NEW] Dependency Injection
        self.on_close_callback = on_close_callback
        
        # Data State
        self.df_raw = None
        self.df = None
        self.time_index = None

        self.audio_len_sec = 0
        self.notes_content = ""
        self.notes_path = None

        self.is_playing = False

        # Timing
        self.start_time = 0
        self.playback_offset = 0
        
        self.is_dragging = False
        self.bg = None

        # Band/Line Toggles
        self.band_vars = {
            'GSR': BooleanVar(value=True),
            'HR': BooleanVar(value=True),
            'HRV': BooleanVar(value=True)
        }
        
        self.lines = {}
        
        # Build UI
        self.setup_gui()

    def setup_gui(self):
        # --- Toolbar ---
        toolbar = tk.Frame(self.master, height=50, bg='#333333')
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)

        # Back / Close Button
        tk.Button(toolbar, text="< Back to Live", command=self.request_close, 
                  bg='#555', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=10, pady=5)

        tk.Button(toolbar, text="Load Session", command=self.load_session, 
                  bg='#004488', fg='white').pack(side=tk.LEFT, padx=5)
        
        self.lbl_file = tk.Label(toolbar, text="No File Loaded", bg='#333', fg='#ccc')
        self.lbl_file.pack(side=tk.LEFT, padx=5)

        tk.Frame(toolbar, width=20, bg='#333').pack(side=tk.LEFT)

        self.btn_play = tk.Button(toolbar, text="Play", command=self.toggle_play, state=tk.DISABLED, width=10, bg='#006600', fg='white')
        self.btn_play.pack(side=tk.LEFT, padx=5)

        # Seek Slider
        self.slider = tk.Scale(toolbar, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0,
                               resolution=0.1, length=400, bg='#333', fg='white', troughcolor='#555')
        self.slider.pack(side=tk.LEFT, padx=5)

        # Bindings for Manual Seek
        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        self.lbl_time = tk.Label(toolbar, text="00:00 / 00:00", bg='#333', fg='cyan', font=('Courier', 12, 'bold'))
        self.lbl_time.pack(side=tk.LEFT, padx=5)

        # --- Smoothing Slider ---
        tk.Frame(toolbar, width=20, bg='#333').pack(side=tk.LEFT)
        tk.Label(toolbar, text="Smooth:", bg='#333', fg='#aaa').pack(side=tk.LEFT)
        self.scale_smooth = tk.Scale(toolbar, from_=1, to=50, orient=tk.HORIZONTAL, length=100, bg='#333', fg='white', troughcolor='#555')
        self.scale_smooth.set(DEFAULT_SMOOTHING)
        self.scale_smooth.pack(side=tk.LEFT, padx=5)
        self.scale_smooth.bind("<ButtonRelease-1>", self.on_smooth_change)

        # --- Band Toggles ---
        toggle_frame = tk.Frame(self.master, bg='#222', bd=1, relief=tk.SUNKEN)
        toggle_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)
        tk.Label(toggle_frame, text="Signals:", bg='#222', fg='#ccc', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)

        colors = {'GSR': 'magenta', 'HR': 'red', 'HRV': 'cyan', 'Delta_TA': 'orange'}
        self.colors = colors
        
        for b in ['GSR', 'HR', 'HRV']:
            cb = Checkbutton(toggle_frame, text=b, variable=self.band_vars[b],
                             bg='#222', fg=colors.get(b, 'white'), selectcolor='#444',
                             activebackground='#333', activeforeground=colors.get(b, 'white'),
                             command=self.refresh_visibility)
            cb.pack(side=tk.LEFT, padx=5)

        # --- Main Area ---
        panes = tk.PanedWindow(self.master, orient=tk.HORIZONTAL, bg='#222')
        panes.pack(fill=tk.BOTH, expand=True)

        # Left: Graph Container
        self.graph_container = tk.Frame(panes, bg='black')
        panes.add(self.graph_container, minsize=800, stretch="always")

        # Y-Scale Slider (Vertical)
        self.scale_y = tk.Scale(self.graph_container, from_=100, to=1, orient=tk.VERTICAL,
                                length=500, label="Scale", bg='#222', fg='white', troughcolor='#444')
        self.scale_y.set(DEFAULT_Y_SCALE)
        self.scale_y.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        self.scale_y.bind("<ButtonRelease-1>", self.on_yscale_change)

        # Matplotlib Area
        self.graph_frame = tk.Frame(self.graph_container, bg='black')
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.graph_frame.bind("<Configure>", self.on_resize)

        # Right: Info/Notes
        self.info_frame = tk.Frame(panes, width=300, bg='#333')
        panes.add(self.info_frame, minsize=300, stretch="never")

        # Notes
        tk.Label(self.info_frame, text="Session Notes", font=('Arial', 12, 'bold'), bg='#333', fg='white').pack(pady=10)
        self.txt_notes = scrolledtext.ScrolledText(self.info_frame, height=15, wrap=tk.WORD, bg='#222', fg='white', insertbackground='white')
        self.txt_notes.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.btn_save_notes = tk.Button(self.info_frame, text="Save Notes", command=self.save_notes, bg='#005500', fg='white')
        self.btn_save_notes.pack(fill=tk.X, padx=10, pady=5)

        # Stats
        tk.Label(self.info_frame, text="Statistics", font=('Arial', 12, 'bold'), bg='#333', fg='white').pack(pady=5)
        self.lbl_stats = tk.Label(self.info_frame, text="--", justify=tk.LEFT, bg='#333', fg='#ccc')
        self.lbl_stats.pack(fill=tk.X, padx=10)

        # --- Matplotlib Init ---
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('#111')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.init_plot()
        
    def request_close(self):
        self.stop_playback(reset=True)
        if self.on_close_callback:
            self.on_close_callback()
        else:
            self.master.destroy()

    def log(self, msg):
        print(f"[Viewer] {msg}")

    def save_notes(self):
        if not self.notes_path:
            messagebox.showwarning("Warning", "No notes file loaded.")
            return
        try:
            content = self.txt_notes.get("1.0", tk.END).strip()
            with open(self.notes_path, "w") as f:
                f.write(content)
            messagebox.showinfo("Saved", "Notes updated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save notes: {e}")

    def on_yscale_change(self, event):
        self.init_plot()
        self.update_plot(self.playback_offset, force_draw=True)

    def init_plot(self):
        self.ax.clear()
        
        self.ax.set_facecolor('#111')
        self.ax.spines['bottom'].set_color('#555')
        self.ax.spines['top'].set_color('#555')
        self.ax.spines['left'].set_color('#555')
        self.ax.spines['right'].set_color('#555')
        self.ax.tick_params(axis='x', colors='#888')
        self.ax.tick_params(axis='y', colors='#888')
        self.ax.yaxis.label.set_color('#888')
        self.ax.xaxis.label.set_color('#888')
        self.ax.title.set_color('white')

        y_max = self.scale_y.get()
        self.ax.set_ylim(0, y_max) # User adjustable

        self.ax.set_xlim(-WINDOW_SEC, 0)
        self.ax.grid(True, alpha=0.2, color='#444')
        self.ax.set_xlabel("Time (s relative to Playhead)")

        self.lines = {}
        for b, col in self.colors.items():
            line, = self.ax.plot([], [], label=b, color=col, lw=1.5)
            self.lines[b] = line
            
        legend = self.ax.legend(loc='upper left', facecolor='#222', edgecolor='#444')
        for text in legend.get_texts(): text.set_color('white')
        
        try:
            self.canvas.draw()
            self.bg = self.canvas.copy_from_bbox(self.ax.bbox)
        except: pass

    def on_resize(self, event):
        self.init_plot()

    def refresh_visibility(self):
        for b, line in self.lines.items():
            if b in self.band_vars:
                line.set_visible(self.band_vars[b].get())
        if not self.is_playing:
            self.update_plot(self.playback_offset, force_draw=True)

    def load_session(self):
        # Default to Session_Data
        initial = os.path.join(os.getcwd(), "Session_Data")
        if not os.path.exists(initial): initial = os.getcwd()
        
        path = filedialog.askopenfilename(initialdir=initial,
                                          filetypes=[("CSV Files", "GSR.csv"), ("All CSV", "*.csv")])
        if not path: return

        try:
            self.lbl_file.config(text=os.path.basename(path))
            self.log(f"Loading: {os.path.basename(path)}")

            self.df_raw = pd.read_csv(path)
            self.df_raw.columns = [c.strip() for c in self.df_raw.columns]
            
            # --- START CSV MERGING LOGIC ---
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            
            hrm_path = os.path.join(dirname, "HRM.csv")
            if os.path.exists(hrm_path):
                try:
                    df_hr = pd.read_csv(hrm_path)
                    df_hr.columns = [c.strip() for c in df_hr.columns]
                    
                    if 'Elapsed' in self.df_raw.columns:
                        self.df_raw['Rel_Time'] = pd.to_numeric(self.df_raw['Elapsed'], errors='coerce')
                    
                    if 'Elapsed' in df_hr.columns:
                        df_hr['Rel_Time'] = pd.to_numeric(df_hr['Elapsed'], errors='coerce')
                        
                    # Rename columns to avoid collision
                    df_hr.rename(columns={'HR_BPM': 'HR', 'RMSSD_MS': 'HRV'}, inplace=True)
                    
                    # Merge (tolerant)
                    self.df = pd.merge_asof(self.df_raw.sort_values('Rel_Time'),
                                            df_hr[['Rel_Time', 'HR', 'HRV']].sort_values('Rel_Time'),
                                            on='Rel_Time', direction='nearest', tolerance=0.1)
                except Exception as ex:
                    print(f"Merge Error: {ex}")
                    self.df = self.df_raw.copy()
            else:
                 self.df = self.df_raw.copy()

            # Ensure Rel_Time
            if 'Rel_Time' not in self.df.columns:
                 if 'Elapsed' in self.df.columns:
                      self.df['Rel_Time'] = pd.to_numeric(self.df['Elapsed'], errors='coerce')
                 else:
                      self.df['Rel_Time'] = np.arange(len(self.df)) * (1.0/60.0) 

            # Fix TA
            if 'TA' in self.df.columns:
                 self.df['GSR'] = pd.to_numeric(self.df['TA'], errors='coerce')

            self.apply_smoothing()
            # --- END CSV MERGING LOGIC ---

            # --- START AUDIO LOAD (Delegated) ---
            audio_path = os.path.join(dirname, "Audio.wav")
            if not os.path.exists(audio_path):
                audio_path = os.path.join(dirname, "audio.wav")
            
            if os.path.exists(audio_path):
                success = self.audio_handler.load_for_playback(audio_path)
                if success:
                    # Calculate duration from raw data provided by handler
                    d_data = self.audio_handler.playback_data
                    d_fs = self.audio_handler.playback_fs
                    if len(d_data.shape) > 1:
                         self.audio_len_sec = d_data.shape[0] / d_fs
                    else:
                         self.audio_len_sec = len(d_data) / d_fs
                    self.log(f"Audio Loaded via Handler: {self.audio_len_sec:.2f}s")
                else:
                    self.audio_len_sec = self.df['Rel_Time'].max()
                    self.log("Handler Failed to Load Audio")
            else:
                self.audio_len_sec = self.df['Rel_Time'].max() if not self.df.empty else 10.0
                self.log("No audio found (Silent Mode)")
            # --- END AUDIO LOAD ---

            # Notes
            notes_path = os.path.join(dirname, "notes.txt")
            self.notes_path = notes_path
            self.notes_content = "No notes."
            if os.path.exists(notes_path):
                with open(notes_path, 'r', errors='ignore') as f:
                    self.notes_content = f.read()

            self.txt_notes.delete('1.0', tk.END)
            self.txt_notes.insert(tk.END, self.notes_content)

            self.lbl_stats.config(text=f"Duration: {self.audio_len_sec:.1f}s")

            # Reset Slider
            self.slider.config(to=self.audio_len_sec)
            self.slider.set(0)
            
            self.playback_offset = 0
            self.stop_playback(reset=True)
            self.btn_play.config(state=tk.NORMAL)
            self.init_plot()
            self.update_plot(0, force_draw=True)

        except Exception as e:
            self.log(f"Load Crash: {e}")
            import traceback; traceback.print_exc()

    def on_smooth_change(self, event):
        if self.df_raw is None: return
        self.apply_smoothing()
        self.update_plot(self.playback_offset, force_draw=True)

    def apply_smoothing(self):
        cols_to_smooth = ['GSR', 'HR', 'HRV']
        win = self.scale_smooth.get()
        if win > 1:
            for c in cols_to_smooth:
                if c in self.df.columns:
                    self.df[c] = self.df[c].rolling(window=win, min_periods=1).mean()
        
        self.time_index = self.df['Rel_Time'].values

    def toggle_play(self):
        if self.is_playing:
            # Stop stream, keep offset
            elapsed = time.time() - self.start_time
            self.playback_offset += elapsed
            self.stop_playback(reset=False)
        else:
            self.start_playback()

    def start_playback(self):
        if self.is_playing: return
        try:
            self.is_playing = True
            self.btn_play.config(text="Pause", bg='#aa0000')

            # --- START PLAYBACK (Delegated) ---
            # Handler's start_playback takes offset in seconds
            self.audio_handler.start_playback(offset=self.playback_offset)
            # --- END PLAYBACK ---
            
            self.start_time = time.time()
            self.animate()

        except Exception as e:
            self.log(f"Start Error: {e}")
            self.stop_playback()

    def stop_playback(self, reset=True):
        self.is_playing = False
        self.btn_play.config(text="Play", bg='#006600')

        try:
            self.audio_handler.stop_playback() # [MOD] Delegated
        except: pass

        if reset:
            self.playback_offset = 0
            self.slider.set(0)
            self.update_plot(0, force_draw=True)

    def on_slider_press(self, event):
        self.is_dragging = True

    def on_slider_release(self, event):
        self.is_dragging = False
        val = self.slider.get()
        self.perform_seek(val)

    def perform_seek(self, val):
        was_playing = self.is_playing
        if was_playing:
            self.stop_playback(reset=False)

        self.playback_offset = float(val)
        self.update_plot(self.playback_offset, force_draw=True)

        if was_playing:
            self.start_playback()

    def animate(self):
        if not self.is_playing: return

        elapsed = time.time() - self.start_time
        current_time = self.playback_offset + elapsed

        if current_time >= self.audio_len_sec:
            self.stop_playback(reset=True)
            return

        if not self.is_dragging:
            self.slider.set(current_time)

        self.update_plot(current_time)
        self.master.after(UPDATE_INTERVAL_MS, self.animate)

    def update_plot(self, current_time, force_draw=False):
        t_str = f"{int(current_time) // 60:02}:{int(current_time) % 60:02}"
        tot_str = f"{int(self.audio_len_sec) // 60:02}:{int(self.audio_len_sec) % 60:02}"
        self.lbl_time.config(text=f"{t_str} / {tot_str}")

        if self.df is None or self.time_index is None: return

        # Get Window
        win_start = current_time - WINDOW_SEC
        idx_start = np.searchsorted(self.time_index, win_start, side='left')
        idx_end = np.searchsorted(self.time_index, current_time, side='right')

        view = self.df.iloc[idx_start:idx_end]
        
        x_data = []
        if not view.empty:
            x_data = view['Rel_Time'].values - current_time # 0 at right edge

        for b, line in self.lines.items():
            if line.get_visible():
                if view.empty or b not in view.columns:
                    line.set_data([], [])
                else:
                    line.set_data(x_data, view[b].values)

        # Blitting
        if self.bg and not force_draw:
            try:
                self.canvas.restore_region(self.bg)
                for line in self.lines.values():
                    if line.get_visible(): self.ax.draw_artist(line)
                self.canvas.blit(self.ax.bbox)
            except:
                self.canvas.draw()
        else:
            self.canvas.draw()
