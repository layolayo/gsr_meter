import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import time

# --- SETTINGS ---
WINDOW_PAST = 7.0
WINDOW_FUTURE = 3.0
UPDATE_INTERVAL_MS = 50  # 20 FPS

class SessionViewer:
    def __init__(self, master_frame, audio_handler, on_close_callback=None):
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
        
        # Rendering
        self.line_gsr = None
        self.line_ta_set = None 
        self.lines_grid = []    
        
        self.mini_line = None
        self.mini_cursor = None
        self.timer_id = None
        
        # UI Vars
        self.var_track = tk.BooleanVar(value=False) 
        
        # Build UI
        self.setup_gui()

    def setup_gui(self):
        # --- Toolbar (Top) ---
        toolbar = tk.Frame(self.master, height=40, bg='#333333')
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)

        # Back
        tk.Button(toolbar, text="< Back", command=self.request_close, 
                  bg='#555', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=10, pady=5)

        # Load
        tk.Button(toolbar, text="Load GSR.csv", command=self.load_session, 
                  bg='#004488', fg='white').pack(side=tk.LEFT, padx=5)
        
        self.lbl_file = tk.Label(toolbar, text="No File Loaded", bg='#333', fg='#ccc')
        self.lbl_file.pack(side=tk.LEFT, padx=5)

        # Controls
        tk.Frame(toolbar, width=20, bg='#333').pack(side=tk.LEFT)
        
        self.btn_play = tk.Button(toolbar, text="Play", command=self.toggle_play, state=tk.DISABLED, width=8, bg='#006600', fg='white')
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.lbl_time = tk.Label(toolbar, text="00:00 / 00:00", bg='#333', fg='cyan', font=('Courier', 12, 'bold'))
        self.lbl_time.pack(side=tk.LEFT, padx=5)

        # Phase 2 Controls
        tk.Frame(toolbar, width=20, bg='#333').pack(side=tk.LEFT)
        
        tk.Checkbutton(toolbar, text="Auto-Track", variable=self.var_track, 
                       bg='#333', fg='white', selectcolor='#444', activeforeground='white', activebackground='#333',
                       command=lambda: self.update_plot(self.playback_offset)).pack(side=tk.LEFT, padx=5)
                       
        tk.Label(toolbar, text="Zoom:", bg='#333', fg='#aaa').pack(side=tk.LEFT)
        # [MOD] Increased range to 10.0
        self.scale_sens = tk.Scale(toolbar, from_=0.2, to=10.0, orient=tk.HORIZONTAL, length=100, resolution=0.1,
                                   bg='#333', fg='white', troughcolor='#555', showvalue=0)
        self.scale_sens.set(1.0) 
        self.scale_sens.pack(side=tk.LEFT, padx=5)
        self.scale_sens.bind("<ButtonRelease-1>", lambda e: self.update_plot(self.playback_offset))


        # --- Main Layout ---
        main_content = tk.Frame(self.master, bg='#222')
        main_content.pack(fill=tk.BOTH, expand=True)
        
        left_panel = tk.Frame(main_content, bg='black')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(main_content, width=250, bg='#333')
        right_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        self.graph_frame = tk.Frame(left_panel, bg='black')
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True) 
        
        slider_frame = tk.Frame(left_panel, bg='#222', height=40)
        slider_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Label(slider_frame, text="Audio/Time:", bg='#222', fg='#aaa', font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        
        self.slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0,
                               resolution=0.1, bg='#222', fg='white', troughcolor='#444', 
                               activebackground='#004488', highlightthickness=0)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)
        
        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)
        # [NEW] Real-time drag update
        self.slider.config(command=self.on_slider_drag)

        tk.Label(right_panel, text="Session Notes", font=('Arial', 10, 'bold'), bg='#333', fg='white').pack(pady=5)
        self.txt_notes = scrolledtext.ScrolledText(right_panel, height=20, width=30, wrap=tk.WORD, bg='#222', fg='white', insertbackground='white')
        self.txt_notes.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.btn_save_notes = tk.Button(right_panel, text="Save Notes", command=self.save_notes, bg='#005500', fg='white')
        self.btn_save_notes.pack(fill=tk.X, padx=5, pady=5)
        
        self.fig = plt.figure(figsize=(10, 8), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')
        
        gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.3)
        self.ax = self.fig.add_subplot(gs[0])      
        self.ax_mini = self.fig.add_subplot(gs[1]) 
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.graph_frame.bind("<Configure>", self.on_resize) 
        
        # [NEW] Interactive Minimap
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        self.init_plot()

    def request_close(self):
        self.stop_playback(reset=True)
        if self.timer_id:
            try: self.master.after_cancel(self.timer_id)
            except: pass
            self.timer_id = None
            
        if self.on_close_callback:
            self.on_close_callback()
        else:
            self.master.destroy() 

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
        # --- 1. Main Graph ---
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_zorder(100)
            
        self.ax.tick_params(axis='x', colors='lightgray')
        self.ax.tick_params(axis='y', colors='lightgray')
        
        self.ax.set_xlim(-WINDOW_PAST, WINDOW_FUTURE)
        self.ax.set_ylim(0, 6.5) 
        
        self.ax.set_title("GSR Zoomed (-7s to +3s)", fontsize=12, fontweight='bold', color='white')
        
        ticks = list(range(int(-WINDOW_PAST), int(WINDOW_FUTURE)+1))
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels([str(x) if x != 0 else "NOW" for x in ticks])

        self.ax.grid(True, which='major', color='#222', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax.axvline(x=0, color='white', linestyle='--', alpha=0.5, zorder=90)

        self.line_ta_set, = self.ax.plot([], [], linestyle='--', color='#CC5500', linewidth=1.5, alpha=0.8, zorder=40, label='Set')

        self.lines_grid = []
        for _ in range(5): 
             l, = self.ax.plot([], [], linestyle=':', color='#CC5500', linewidth=1.0, alpha=0.5, zorder=30)
             self.lines_grid.append(l)

        self.line_gsr, = self.ax.plot([], [], lw=2, color='magenta', label='GSR', zorder=50)

        # --- 2. Minimap ---
        self.ax_mini.clear()
        self.ax_mini.set_facecolor('#111111') 
        
        for spine in self.ax_mini.spines.values():
            spine.set_edgecolor('#444444')
            
        self.ax_mini.tick_params(axis='x', colors='gray', labelsize=8)
        self.ax_mini.tick_params(axis='y', colors='gray', labelsize=8)
        self.ax_mini.set_title("Full Session Overlay (Click to Seek)", fontsize=10, color='gray') # [MOD] Title hint
        
        if self.df is not None and not self.df.empty:
            t = self.df['Rel_Time'].values
            y = self.df['GSR'].values
            self.ax_mini.plot(t, y, color='cyan', lw=1, alpha=0.6)
            self.ax_mini.set_xlim(t.min(), t.max())
            
            y_min, y_max = y.min(), y.max()
            margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.5
            self.ax_mini.set_ylim(max(0, y_min - margin), y_max + margin)
            
        self.mini_cursor = self.ax_mini.axvline(x=0, color='yellow', lw=1.5, alpha=0.9)

        try:
             # self.fig.tight_layout() 
             pass
             self.canvas.draw()
        except: pass

    def on_resize(self, event):
        self.init_plot()
        if self.df is not None:
             self.update_plot(self.playback_offset)

    def on_plot_click(self, event):
        # [NEW] Handle click on Minimap
        if event.inaxes == self.ax_mini:
             x_val = event.xdata
             if x_val is not None:
                 self.slider.set(x_val)
                 self.perform_seek(x_val)

    def load_session(self):
        initial = os.path.join(os.getcwd(), "Session_Data")
        if not os.path.exists(initial): initial = os.getcwd()
        
        path = filedialog.askopenfilename(initialdir=initial, title="Select GSR Recording",
                                          filetypes=[("GSR CSV", "GSR.csv"), ("All CSV", "*.csv")])
        if not path: return
        
        folder_path = os.path.dirname(path)
        gsr_path = path

        try:
            self.lbl_file.config(text=os.path.basename(folder_path))
            self.log(f"Loading Session: {folder_path}")

            self.df = pd.read_csv(gsr_path, low_memory=False)
            self.df.columns = [c.strip() for c in self.df.columns]
            
            if 'Rel_Time' not in self.df.columns:
                if 'Elapsed' in self.df.columns:
                    self.df['Rel_Time'] = pd.to_numeric(self.df['Elapsed'], errors='coerce')
                else:
                    self.df['Rel_Time'] = np.arange(len(self.df)) * (1.0/60.0)
            
            if 'GSR' not in self.df.columns and 'TA' in self.df.columns:
                 self.df['GSR'] = pd.to_numeric(self.df['TA'], errors='coerce')
            
            # [NEW] Fix Missing Metadata (ffill/bfill)
            # Ensure columns exist first
            cols_to_fix = ['Center', 'TA SET', 'Window', 'Win']
            for c in cols_to_fix:
                if c in self.df.columns:
                    self.df[c] = self.df[c].ffill().bfill()
            
            self.time_index = self.df['Rel_Time'].values

            # Audio
            audio_path = os.path.join(folder_path, "Audio.wav")
            if not os.path.exists(audio_path): audio_path = os.path.join(folder_path, "audio.wav")
            
            if os.path.exists(audio_path):
                success = self.audio_handler.load_for_playback(audio_path)
                if success:
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
                self.audio_len_sec = self.df['Rel_Time'].max() if not self.df.empty else 10.0
                self.log("No Audio File")

            notes_path = os.path.join(folder_path, "notes.txt")
            self.notes_path = notes_path
            content = "No notes."
            if os.path.exists(notes_path):
                try:
                    with open(notes_path, 'r', errors='ignore') as f: content = f.read()
                except: pass
            
            self.txt_notes.delete('1.0', tk.END)
            self.txt_notes.insert(tk.END, content)

            self.slider.config(to=self.audio_len_sec)
            self.slider.set(0)
            self.playback_offset = 0
            self.init_plot()
            
            self.btn_play.config(state=tk.NORMAL)
            self.stop_playback(reset=True)

        except Exception as e:
            self.log(f"Load Error: {e}")
            import traceback; traceback.print_exc()

    def toggle_play(self):
        if self.is_playing:
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
            self.audio_handler.start_playback(offset=self.playback_offset)
            self.start_time = time.time()
            self.animate()
        except Exception as e:
            self.log(f"Start Error: {e}")
            self.stop_playback()

    def stop_playback(self, reset=True):
        self.is_playing = False
        self.btn_play.config(text="Play", bg='#006600')
        
        if self.timer_id:
             try: self.master.after_cancel(self.timer_id)
             except: pass
             self.timer_id = None
        
        try: self.audio_handler.stop_playback()
        except: pass
        if reset:
            self.playback_offset = 0
            self.slider.set(0)
            self.update_plot(0)

    def on_slider_press(self, event):
        self.is_dragging = True

    def on_slider_release(self, event):
        self.is_dragging = False
        val = self.slider.get()
        self.perform_seek(val)

    def on_slider_drag(self, val):
        # [NEW] Real-time updates for Minimap Cursor
        # Avoid full redraw of main graph if possible?
        # For now, just call update_plot, performance should be ok.
        if self.is_dragging:
             self.update_plot(float(val))

    def perform_seek(self, val):
        was_playing = self.is_playing
        if was_playing:
            self.stop_playback(reset=False)
        self.playback_offset = float(val)
        self.update_plot(self.playback_offset)
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
        if self.is_playing:
             self.timer_id = self.master.after(UPDATE_INTERVAL_MS, self.animate)
        else:
             self.timer_id = None

    def update_plot(self, current_time):
        t_str = f"{int(current_time) // 60:02}:{int(current_time) % 60:02}"
        tot_str = f"{int(self.audio_len_sec) // 60:02}:{int(self.audio_len_sec) % 60:02}"
        self.lbl_time.config(text=f"{t_str} / {tot_str}")

        if self.df is None or self.time_index is None: return

        # 1. Update Zoomed Graph
        win_start = current_time - WINDOW_PAST
        win_end = current_time + WINDOW_FUTURE
        
        idx_start = np.searchsorted(self.time_index, win_start, side='left')
        idx_end = np.searchsorted(self.time_index, win_end, side='right')
        
        view = self.df.iloc[idx_start:idx_end]
        
        idx_now = np.searchsorted(self.time_index, current_time)
        c_val = 3.0
        w_val = 1.0
        has_meta = False
        
        if idx_now < len(self.df):
            row = self.df.iloc[idx_now]
            if 'Center' in row: 
                c_val = row['Center']; has_meta = True
            elif 'TA SET' in row:
                c_val = row['TA SET']; has_meta = True
            
            if 'Window' in row: 
                w_val = row['Window']; has_meta = True
            elif 'Win' in row:
                w_val = row['Win']; has_meta = True

        sens_mult = self.scale_sens.get()
        eff_win = w_val / sens_mult
        
        unit_val = 0.2 * eff_win
        
        if view.empty:
            self.line_gsr.set_data([], [])
        else:
            x_data = view['Rel_Time'].values - current_time 
            y_data = view['GSR'].values
            self.line_gsr.set_data(x_data, y_data)
            
            if self.var_track.get():
                if len(y_data) > 1:
                    ymin, ymax = y_data.min(), y_data.max()
                    pad = 0.1
                    self.ax.set_ylim(ymin - pad, ymax + pad)
            else:
                min_p = c_val - (0.625 * eff_win)
                max_p = c_val + (0.375 * eff_win)
                self.ax.set_ylim(min_p, max_p)

        x_span = np.array([-WINDOW_PAST, WINDOW_FUTURE])
        
        if self.line_ta_set:
            self.line_ta_set.set_data(x_span, [c_val, c_val])
            
        unit_mults = [2, 1, -1, -2, -3]
        for i, l in enumerate(self.lines_grid):
            if i < len(unit_mults):
                u_y = c_val + (unit_mults[i] * unit_val)
                l.set_data(x_span, [u_y, u_y])

        # 2. Minimap
        if self.mini_cursor:
            self.mini_cursor.set_xdata([current_time])

        # 3. Draw
        try:
            self.canvas.draw()
        except: pass
