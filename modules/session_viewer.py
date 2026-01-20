import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import time

# --- SETTINGS ---
WINDOW_SEC = 10.0  # Matches v42 Main Graph
UPDATE_INTERVAL_MS = 50  # 20 FPS

class SessionViewer:
    def __init__(self, master_frame, audio_handler, on_close_callback=None):
        """
        master_frame: Parent widget.
        audio_handler: Dependency injected AudioHandler.
        on_close_callback: Function to call on close.
        """
        self.master = master_frame
        self.audio_handler = audio_handler
        self.on_close_callback = on_close_callback
        
        # Data State
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
        
        # Blit Background
        self.bg = None
        self.line_gsr = None
        
        # Build UI
        self.setup_gui()

    def setup_gui(self):
        # --- Toolbar (Top) ---
        toolbar = tk.Frame(self.master, height=40, bg='#333333')
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)

        # Back Button
        tk.Button(toolbar, text="< Back", command=self.request_close, 
                  bg='#555', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=10, pady=5)

        # Load Folder Button
        tk.Button(toolbar, text="Load Session Folder", command=self.load_session, 
                  bg='#004488', fg='white').pack(side=tk.LEFT, padx=5)
        
        self.lbl_file = tk.Label(toolbar, text="No Folder Loaded", bg='#333', fg='#ccc')
        self.lbl_file.pack(side=tk.LEFT, padx=5)

        # Play / Time Controls
        tk.Frame(toolbar, width=30, bg='#333').pack(side=tk.LEFT)
        
        self.btn_play = tk.Button(toolbar, text="Play", command=self.toggle_play, state=tk.DISABLED, width=10, bg='#006600', fg='white')
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.lbl_time = tk.Label(toolbar, text="00:00 / 00:00", bg='#333', fg='cyan', font=('Courier', 12, 'bold'))
        self.lbl_time.pack(side=tk.LEFT, padx=5)

        # --- Main Layout (Using PanedWindow or just Pack) ---
        # User asked for slider UNDERNEATH graph.
        
        main_content = tk.Frame(self.master, bg='#222')
        main_content.pack(fill=tk.BOTH, expand=True)
        
        # Left Side: Graph + Slider
        left_panel = tk.Frame(main_content, bg='black')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Side: Notes (Optional but useful)
        right_panel = tk.Frame(main_content, width=250, bg='#333')
        right_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        # --- Graph Area ---
        self.graph_frame = tk.Frame(left_panel, bg='black')
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Takes most space
        
        # --- Slider Area (Immediately Below Graph) ---
        slider_frame = tk.Frame(left_panel, bg='#222', height=40)
        slider_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Label(slider_frame, text="Audio/Time:", bg='#222', fg='#aaa', font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        
        self.slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0,
                               resolution=0.1, bg='#222', fg='white', troughcolor='#444', 
                               activebackground='#004488', highlightthickness=0)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)
        
        # Bindings for Seek
        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        # --- Notes Area ---
        tk.Label(right_panel, text="Session Notes", font=('Arial', 10, 'bold'), bg='#333', fg='white').pack(pady=5)
        self.txt_notes = scrolledtext.ScrolledText(right_panel, height=20, width=30, wrap=tk.WORD, bg='#222', fg='white', insertbackground='white')
        self.txt_notes.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.btn_save_notes = tk.Button(right_panel, text="Save Notes", command=self.save_notes, bg='#005500', fg='white')
        self.btn_save_notes.pack(fill=tk.X, padx=5, pady=5)
        
        # --- Matplotlib Init ---
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e') # Matches v42 bg
        self.ax.set_facecolor('#1e1e1e')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.graph_frame.bind("<Configure>", self.on_resize) # Handle resize

        self.init_plot()

    def request_close(self):
        self.stop_playback(reset=True)
        if self.on_close_callback:
            self.on_close_callback()
        else:
            self.master.destroy() # Fallback

    def log(self, msg):
        print(f"[Viewer] {msg}")

    def save_notes(self):
        if not self.notes_path:
            messagebox.showwarning("Warning", "No notes loaded.")
            return
        try:
            content = self.txt_notes.get("1.0", tk.END).strip()
            with open(self.notes_path, "w") as f:
                f.write(content)
            messagebox.showinfo("Saved", "Notes updated.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def init_plot(self):
        self.ax.clear()
        
        # v42 Aesthetics
        self.ax.set_facecolor('#1e1e1e')
        
        # Spines
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_zorder(100) # On top
            
        self.ax.tick_params(axis='x', colors='lightgray')
        self.ax.tick_params(axis='y', colors='lightgray')
        
        # Fixed limits as requested ("Replica of v42")
        self.ax.set_ylim(-5, 105) 
        self.ax.set_xlim(-WINDOW_SEC, 0)
        
        # Labels
        self.ax.set_title("GSR Monitor (Recorded)", fontsize=14, fontweight='bold', color='white')
        # X-Axis Labels 
        self.ax.set_xticks([x for x in range(int(-WINDOW_SEC), 1)])
        self.ax.set_xticklabels([str(abs(x)) for x in range(int(-WINDOW_SEC), 1)])

        # --- v42 Custom Grid Lines ---
        # TA Set Line
        self.ax.axhline(y=62.5, color='#CC5500', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
        
        # Unit Grid Lines (+/- 20 Units)
        unit_defs = [102.5, 82.5, 42.5, 22.5, 2.5]
        for y_u in unit_defs:
             self.ax.axhline(y=y_u, color='#CC5500', linestyle=':', linewidth=1.0, alpha=0.5, zorder=0)

        # Plot Line (Magenta, v42 style)
        self.line_gsr, = self.ax.plot([], [], lw=2, color='magenta', label='GSR', zorder=50)

        self.fig.tight_layout()
        
        try:
            self.canvas.draw()
            self.bg = self.canvas.copy_from_bbox(self.ax.bbox)
        except: pass

    def on_resize(self, event):
        self.init_plot()
        # Redraw current frame if valid
        if self.df is not None:
            self.update_plot(self.playback_offset, force_draw=True)

    def load_session(self):
        # 1. Ask for Directory
        initial = os.path.join(os.getcwd(), "Session_Data")
        if not os.path.exists(initial): initial = os.getcwd()
        
        folder_path = filedialog.askdirectory(initialdir=initial, title="Select Session Folder")
        if not folder_path: return

        # 2. Look for GSR.csv
        gsr_path = os.path.join(folder_path, "GSR.csv")
        if not os.path.exists(gsr_path):
            messagebox.showerror("Error", "GSR.csv not found in selected folder.")
            return
            
        try:
            self.lbl_file.config(text=os.path.basename(folder_path))
            self.log(f"Loading Session: {folder_path}")

            # Read CSV
            self.df = pd.read_csv(gsr_path, low_memory=False)
            self.df.columns = [c.strip() for c in self.df.columns]
            
            # Ensure Time Column
            if 'Rel_Time' not in self.df.columns:
                if 'Elapsed' in self.df.columns:
                    self.df['Rel_Time'] = pd.to_numeric(self.df['Elapsed'], errors='coerce')
                else:
                    # Fallback if no time column (assume 60Hz)
                    self.df['Rel_Time'] = np.arange(len(self.df)) * (1.0/60.0)
            
            # Ensure GSR Value
            if 'GSR' not in self.df.columns and 'TA' in self.df.columns:
                 self.df['GSR'] = pd.to_numeric(self.df['TA'], errors='coerce')
            
            # Create Index for Fast Lookup
            self.time_index = self.df['Rel_Time'].values

            # --- Audio Loading ---
            # Try Audio.wav or audio.wav
            audio_path = os.path.join(folder_path, "Audio.wav")
            if not os.path.exists(audio_path): audio_path = os.path.join(folder_path, "audio.wav")
            
            if os.path.exists(audio_path):
                success = self.audio_handler.load_for_playback(audio_path)
                if success:
                     # Get Duration
                     d_data = self.audio_handler.playback_data
                     d_fs = self.audio_handler.playback_fs
                     if len(d_data.shape) > 1: dur = d_data.shape[0] / d_fs
                     else: dur = len(d_data) / d_fs
                     self.audio_len_sec = dur
                     self.log(f"Audio Loaded: {dur:.1f}s")
                else:
                     self.audio_len_sec = self.df['Rel_Time'].max()
                     self.log("Audio Load Failed")
            else:
                self.audio_len_sec = self.df['Rel_Time'].max()
                self.log("No Audio File")

            # --- Notes Loading ---
            notes_path = os.path.join(folder_path, "notes.txt")
            self.notes_path = notes_path
            content = "No notes."
            if os.path.exists(notes_path):
                try:
                    with open(notes_path, 'r', errors='ignore') as f: content = f.read()
                except: pass
            
            self.txt_notes.delete('1.0', tk.END)
            self.txt_notes.insert(tk.END, content)

            # --- Reset UI ---
            self.slider.config(to=self.audio_len_sec)
            self.slider.set(0)
            self.playback_offset = 0
            
            self.btn_play.config(state=tk.NORMAL)
            self.stop_playback(reset=True)

        except Exception as e:
            self.log(f"Load Error: {e}")
            import traceback; traceback.print_exc()

    def toggle_play(self):
        if self.is_playing:
            # Pause
            elapsed = time.time() - self.start_time
            self.playback_offset += elapsed
            self.stop_playback(reset=False)
        else:
            self.start_playback()

    def start_playback(self):
        if self.is_playing: return
        try:
            self.is_playing = True
            self.btn_play.config(text="Pause", bg='#aa0000') # Red for Stop/Pause
            
            # Start Audio
            self.audio_handler.start_playback(offset=self.playback_offset)
            
            self.start_time = time.time()
            self.animate()
        except Exception as e:
            self.log(f"Start Error: {e}")
            self.stop_playback()

    def stop_playback(self, reset=True):
        self.is_playing = False
        self.btn_play.config(text="Play", bg='#006600')
        
        try: self.audio_handler.stop_playback()
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
        # Update Time Label
        t_str = f"{int(current_time) // 60:02}:{int(current_time) % 60:02}"
        tot_str = f"{int(self.audio_len_sec) // 60:02}:{int(self.audio_len_sec) % 60:02}"
        self.lbl_time.config(text=f"{t_str} / {tot_str}")

        if self.df is None or self.time_index is None: return

        # Get Data Window [Current - 10s : Current]
        win_start = current_time - WINDOW_SEC
        
        # Binary Search for indices
        idx_start = np.searchsorted(self.time_index, win_start, side='left')
        idx_end = np.searchsorted(self.time_index, current_time, side='right')
        
        view = self.df.iloc[idx_start:idx_end]
        
        if view.empty:
            self.line_gsr.set_data([], [])
        else:
            # X values should be relative to right edge (0) -> -10 to 0
            # Data Time - Current Time
            x_data = view['Rel_Time'].values - current_time
            y_data = view['GSR'].values
            self.line_gsr.set_data(x_data, y_data)

        # Blit
        if self.bg and not force_draw:
            try:
                self.canvas.restore_region(self.bg)
                self.ax.draw_artist(self.line_gsr)
                self.canvas.blit(self.ax.bbox)
            except:
                self.canvas.draw() # Fallback
        else:
            self.canvas.draw()
