import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import time

# --- SETTINGS ---
import math
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
        self.pattern_index = [] # [NEW] Stores (pattern, start_time) tuples

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
        
        # [NEW] Smoothing State
        self.smooth_c_val = None
        self.smooth_w_val = None
        self.last_plot_time = -1.0
        
        # [NEW] Pattern Logic State
        self.patterns = ["LONG FALL", "BLOWDOWN", "LONG RISE", "ROCKET READ"]
        self.pattern_colors = {
            "LONG FALL": "#008000",
            "BLOWDOWN": "#00CED1",
            "LONG RISE": "#FF4500",
            "ROCKET READ": "#DC143C"
        }
        self.pattern_vars = {} # Will hold BooleanVars
        self.pattern_checkboxes = {} # [NEW] Stores Checkbutton widgets
        self.mini_markers = [] # Store matplotlib lines
        
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
        
        # [MOD] Play button and Timer moved to bottom


        # --- Main Layout ---
        main_content = tk.Frame(self.master, bg='#222')
        main_content.pack(fill=tk.BOTH, expand=True)
        
        left_panel = tk.Frame(main_content, bg='#222') 
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(main_content, width=250, bg='#333')
        right_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        self.graph_frame = tk.Frame(left_panel, bg='#222') # [MOD] Matches panel background
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True) 
        

        tk.Label(right_panel, text="Session Notes", font=('Arial', 10, 'bold'), bg='#333', fg='white').pack(pady=5)
        self.txt_notes = scrolledtext.ScrolledText(right_panel, width=30, wrap=tk.WORD, bg='#222', fg='white', insertbackground='white')
        self.txt_notes.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.btn_save_notes = tk.Button(right_panel, text="Save Notes", command=self.save_notes, bg='#005500', fg='white')
        self.btn_save_notes.pack(fill=tk.X, padx=5, pady=5)

        # [NEW] Pattern Selection Box (Moved to bottom)
        tk.Frame(right_panel, height=2, bg='#444').pack(fill=tk.X, pady=10)
        tk.Label(right_panel, text="Pattern Highlights", font=('Arial', 10, 'bold'), bg='#333', fg='white').pack(pady=5)
        
        sel_frame = tk.Frame(right_panel, bg='#333')
        sel_frame.pack(fill=tk.X, padx=10, pady=(0, 20)) # Added bottom padding
        
        for p in self.patterns:
            var = tk.BooleanVar(value=False)
            self.pattern_vars[p] = var
            cb = tk.Checkbutton(sel_frame, text=p, variable=var, 
                                bg='#333', fg=self.pattern_colors[p], 
                                activebackground='#333', selectcolor='#222',
                                highlightthickness=0, borderwidth=0,
                                state=tk.DISABLED, # [NEW] Start disabled
                                command=self.update_minimap_markers)
            cb.pack(anchor=tk.W)
            self.pattern_checkboxes[p] = cb # [NEW] Store reference
        
        self.fig = plt.figure(figsize=(10, 8), dpi=100)
        self.fig.patch.set_facecolor('#222222') # [MOD] Matches UI background
        
        gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.3)
        self.ax = self.fig.add_subplot(gs[0])      
        self.ax_mini = self.fig.add_subplot(gs[1])
        
        # [NEW] Tighten margins to bring minimap close to timer
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.graph_frame.bind("<Configure>", self.on_resize) 

        # [NEW] Timer (Directly in graph_frame for tightest packing)
        # Using #222 background to look transparent against the gray panel
        self.lbl_time = tk.Label(self.graph_frame, text="00:00 / 00:00", bg='#222', fg='cyan', font=('Courier', 16, 'bold'))
        self.lbl_time.pack(side=tk.BOTTOM, pady=(0, 5)) 

        # [NEW] Navigation Buttons Frame
        # Reordered and color-coded seeking buttons
        seek_frame = tk.Frame(left_panel, bg='#222') # [MOD] Removed height to let content pack tightly
        seek_frame.pack(side=tk.TOP, fill=tk.X)
        
        inner_seek = tk.Frame(seek_frame, bg='#222')
        inner_seek.pack(expand=True)

        # Pattern Selection Buttons (Dark Blue)
        tk.Button(inner_seek, text="|< PREV PAT", command=lambda: self.seek_to_pattern(-1),
                  bg='#003366', fg='white', width=12).pack(side=tk.LEFT, padx=5)

        # Seconds Changing Buttons (Dark Gray)
        tk.Button(inner_seek, text="<< -5sec", command=self.seek_backward,
                  bg='#444', fg='white', width=10).pack(side=tk.LEFT, padx=5)
        
        # [NEW] Play button centered
        self.btn_play = tk.Button(inner_seek, text="Play", command=self.toggle_play, state=tk.DISABLED, 
                                 width=15, bg='#006600', fg='white', font=('Arial', 10, 'bold'))
        self.btn_play.pack(side=tk.LEFT, padx=5)

        tk.Button(inner_seek, text="+ 5sec >>", command=self.seek_forward,
                  bg='#444', fg='white', width=10).pack(side=tk.LEFT, padx=5)

        # Pattern Selection Buttons (Dark Blue)
        tk.Button(inner_seek, text="NEXT PAT >|", command=lambda: self.seek_to_pattern(1),
                  bg='#003366', fg='white', width=12).pack(side=tk.LEFT, padx=5)

        # [NEW] Interactive Minimap
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        self.init_plot()

    def request_close(self):
        """Standard cleanup for component shutdown"""
        self.is_playing = False
        self.playback_offset = 0
        
        try:
            if hasattr(self, 'timer_id') and self.timer_id:
                self.master.after_cancel(self.timer_id)
                self.timer_id = None
        except: pass
        
        try:
            self.stop_playback(reset=True)
        except: pass
        
        # [NEW] Clean up matplotlib resources
        try:
            if hasattr(self, 'fig') and self.fig:
                plt.close(self.fig)
        except: pass
        
        # Call the parent callback to return to main view
        if self.on_close_callback:
            self.on_close_callback()

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
            
        self.ax.get_yaxis().set_visible(False) # [NEW] Hide standard Y-axis
        
        self.ax.set_xlim(-WINDOW_PAST, WINDOW_FUTURE)
        self.ax.set_ylim(-5, 105) # [MOD] Fixed linear display space
        
        self.ax.set_title("EK GSR Session Viewer (Click Minimap to Seek)", fontsize=12, fontweight='bold', color='white')
        
        # [NEW] Relative X-axis labels for Main Plot (-7s to +3s)
        ticks = list(range(int(-WINDOW_PAST), int(WINDOW_FUTURE)+1))
        self.ax.set_xticks(ticks)
        self.ax.tick_params(axis='x', colors='lightgray', labelsize=8)
        self.ax.set_xticklabels([str(x) if x != 0 else "NOW" for x in ticks])

        self.ax.grid(True, which='major', color='#222', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax.axvline(x=0, color='white', linestyle='--', alpha=0.5, zorder=90)

        self.line_ta_set, = self.ax.plot([-WINDOW_PAST, WINDOW_FUTURE], [62.5, 62.5], linestyle='--', color='#CC5500', linewidth=1.5, alpha=0.8, zorder=40)

        self.lines_grid = []
        # Main App Units: +40, +20, -20, -40, -60 from 62.5
        y_offsets = [102.5, 82.5, 42.5, 22.5, 2.5] 
        for y_pos in y_offsets:
             l, = self.ax.plot([-WINDOW_PAST, WINDOW_FUTURE], [y_pos, y_pos], linestyle=':', color='#CC5500', linewidth=1.0, alpha=0.5, zorder=30)
             self.lines_grid.append(l)

        self.line_gsr, = self.ax.plot([], [], lw=2, color='magenta', label='GSR', zorder=50)

        # [NEW] Grid Labels (Docked to Left OUTSIDE at FIXED y-positions)
        self.txt_ta_set_line = self.ax.text(-0.03, 62.5, "", color='#CC5500', fontsize=8, fontweight='bold', 
                                            ha='right', va='top', rotation=45, transform=self.ax.get_yaxis_transform(), clip_on=False)
        self.txt_grid_labels = []
        for y_pos in y_offsets:
            t = self.ax.text(-0.03, y_pos, "", color='#CC5500', fontsize=7, alpha=0.7, 
                             ha='right', va='top', rotation=45, transform=self.ax.get_yaxis_transform(), clip_on=False)
            self.txt_grid_labels.append(t)


        # [NEW] TA Counter Display (Top Left)
        self.txt_ta_counter = self.ax.text(0.02, 0.95, "", color='#aaaaaa', fontsize=9, fontweight='bold',
                                           ha='left', va='top', transform=self.ax.transAxes)
        
        # [NEW] Span Indicator (Top Right)
        self.txt_span = self.ax.text(0.98, 0.95, "", color='#aaaaaa', fontsize=9, fontweight='bold',
                                     ha='right', va='top', transform=self.ax.transAxes)
        
        # [NEW] Pattern Display (Bottom Center, like v42)
        self.txt_pattern = self.ax.text(0.5, 0.02, "", ha='center', va='bottom', 
                                        fontsize=14, fontweight='bold', color='gray', 
                                        transform=self.ax.transAxes, zorder=90)

        # --- 2. Minimap ---
        self.ax_mini.clear()
        self.ax_mini.set_facecolor('#111111') 
        
        for spine in self.ax_mini.spines.values():
            spine.set_edgecolor('#444444')
            
        self.ax_mini.tick_params(axis='x', colors='lightgray', labelsize=8)
        self.ax_mini.tick_params(axis='y', colors='lightgray', labelsize=8)
        self.ax_mini.set_title("Full Session Overlay (Click to Seek)", fontsize=10, color='gray') # [MOD] Title hint
        
        # [NEW] Absolute Time Formatting for Minimap (mm:ss)
        from matplotlib.ticker import FuncFormatter
        def time_formatter(x, pos):
            mins = int(x) // 60
            secs = int(x) % 60
            return f"{mins:02}:{secs:02}"
        self.ax_mini.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        
        
        if self.df is not None and not self.df.empty:
            t = self.df['Rel_Time'].values
            y = self.df['GSR'].values
            
            # [FIX] Filter out NaN and Inf values
            valid_mask = np.isfinite(t) & np.isfinite(y)
            t_valid = t[valid_mask]
            y_valid = y[valid_mask]
            
            if len(t_valid) > 0 and len(y_valid) > 0:
                self.ax_mini.plot(t_valid, y_valid, color='yellow', lw=1, alpha=0.6)
                self.ax_mini.set_xlim(t_valid.min(), t_valid.max())
                
                y_min, y_max = y_valid.min(), y_valid.max()
                margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.5
                self.ax_mini.set_ylim(max(0, y_min - margin), y_max + margin)
            else:
                # Fallback if no valid data
                self.ax_mini.set_xlim(0, 10)
                self.ax_mini.set_ylim(0, 10)
            
        self.mini_cursor = self.ax_mini.axvline(x=0, color='white', lw=1.5, alpha=0.9)

        try:
             # self.fig.tight_layout() 
             pass
             self.canvas.draw()
        except: pass

    def on_resize(self, event):
        self.init_plot()
        if self.df is not None:
             self.update_plot(self.playback_offset)
             self.update_minimap_markers()

    def on_plot_click(self, event):
        # [NEW] Handle click on Minimap
        if event.inaxes == self.ax_mini:
             x_val = event.xdata
             if x_val is not None:
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
            
            # [FIX] Use 'Elapsed' column directly (actual CSV header)
            if 'Elapsed' in self.df.columns:
                # Check first non-null value to detect format
                first_val = self.df['Elapsed'].dropna().iloc[0] if len(self.df['Elapsed'].dropna()) > 0 else None
                is_time_format = False
                
                if first_val is not None:
                    # Check if it's a string with colons (time format)
                    if isinstance(first_val, str) and ':' in first_val:
                        is_time_format = True
                        self.log(f"Detected time format: {first_val}")
                
                if is_time_format:
                    # Parse hh:mm:ss.xxxxxx format
                    def parse_time_to_seconds(time_str):
                        """Convert hh:mm:ss.xxxxxx to total seconds"""
                        try:
                            if pd.isna(time_str):
                                return np.nan
                            parts = str(time_str).split(':')
                            if len(parts) == 3:
                                hours = float(parts[0])
                                minutes = float(parts[1])
                                seconds = float(parts[2])
                                total = hours * 3600 + minutes * 60 + seconds
                                return total
                            else:
                                # Try direct numeric conversion
                                return float(time_str)
                        except Exception as e:
                            self.log(f"Parse error for '{time_str}': {e}")
                            return np.nan
                    
                    self.df['Elapsed'] = self.df['Elapsed'].apply(parse_time_to_seconds)
                    self.log(f"Converted time format. Sample: {self.df['Elapsed'].iloc[0]:.2f}s")
                else:
                    # Already numeric or can be converted directly
                    self.df['Elapsed'] = pd.to_numeric(self.df['Elapsed'], errors='coerce')
                
                # Create Rel_Time as alias for compatibility
                self.df['Rel_Time'] = self.df['Elapsed']
            else:
                # Fallback if no Elapsed column
                self.df['Rel_Time'] = np.arange(len(self.df)) * (1.0/60.0)
                self.df['Elapsed'] = self.df['Rel_Time']
            
            # Map TA to GSR for compatibility
            if 'GSR' not in self.df.columns and 'TA' in self.df.columns:
                 self.df['GSR'] = pd.to_numeric(self.df['TA'], errors='coerce')
            
            # [NEW] Fix Missing Metadata (ffill/bfill)
            cols_to_fix = ['TA SET', 'Sensitivity', 'Window_Size', 'Pivot']
            for c in cols_to_fix:
                if c in self.df.columns:
                    self.df[c] = self.df[c].ffill().bfill()
            
            self.time_index = self.df['Rel_Time'].values

            # [NEW] Pre-calculate Pattern Index for fast navigation
            self.pattern_index = []
            if 'Pattern' in self.df.columns:
                # Find where pattern changes
                mask = (self.df['Pattern'] != self.df['Pattern'].shift(1))
                transitions = self.df[mask]
                for _, row in transitions.iterrows():
                    p_name = row['Pattern']
                    if pd.notna(p_name) and str(p_name).strip() != "":
                        self.pattern_index.append((str(p_name).strip(), float(row['Rel_Time'])))
            
            # [NEW] Enable checkboxes only for patterns found in session
            found_pats = set(p for p, t in self.pattern_index)
            for p_name, cb in self.pattern_checkboxes.items():
                if p_name in found_pats:
                    cb.config(state=tk.NORMAL)
                else:
                    self.pattern_vars[p_name].set(False)
                    cb.config(state=tk.DISABLED)
            
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

            self.playback_offset = 0
            self.init_plot()
            self.update_minimap_markers()
            
            self.btn_play.config(state=tk.NORMAL)
            self.stop_playback(reset=True)
            
            # [NEW] Reset Smoothing on Load
            self.smooth_c_val = None
            self.smooth_w_val = None

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
            self.update_plot(0)

    def on_slider_release(self, event):
        # Slider removed, method kept for compatibility
        pass

    def on_slider_drag(self, val):
        # Slider removed, method kept for compatibility
        pass

    def perform_seek(self, val):
        was_playing = self.is_playing
        if was_playing:
            self.stop_playback(reset=False)
        self.playback_offset = float(max(0, min(val, self.audio_len_sec)))
        self.update_plot(self.playback_offset)
        if was_playing:
            self.start_playback()

    def update_minimap_markers(self):
        """Redraw pattern markers on the minimap using the pre-calculated index"""
        for m in self.mini_markers:
            try: m.remove()
            except: pass
        self.mini_markers = []
        
        if not self.pattern_index: return
            
        selected = [p for p, var in self.pattern_vars.items() if var.get()]
        if not selected:
            try: self.canvas.draw()
            except: pass
            return
            
        for pat, t in self.pattern_index:
            if pat in selected:
                col = self.pattern_colors.get(pat, 'gray')
                l = self.ax_mini.axvline(x=t, color=col, lw=1.5, alpha=0.8, zorder=40)
                self.mini_markers.append(l)
            
        try: self.canvas.draw()
        except: pass

    def seek_to_pattern(self, direction):
        """Jump to the next/prev pattern start using the cached index"""
        if not self.pattern_index: return
        
        selected = [p for p, var in self.pattern_vars.items() if var.get()]
        if not selected:
             messagebox.showinfo("Note", "Select at least one pattern highlight first.")
             return
             
        curr_t = self.get_current_time()
        
        # Filter index for selected types
        valid_starts = [t for pat, t in self.pattern_index if pat in selected]
        
        if not valid_starts:
             messagebox.showinfo("Note", "No instances of selected patterns found.")
             return
             
        target = None
        if direction > 0:
             # Next: Find first start > curr_t + 3.1s (to skip the current 3s lead-in)
             # This ensures we skip the pattern we are currently jump-positioned at.
             for t in valid_starts:
                 if t > curr_t + 3.1: 
                     target = t
                     break
        else:
             # Prev: Find last start < curr_t + 2.9s (to find the start of previous patterns)
             # Looking specifically for patterns before the current logical start window
             for t in reversed(valid_starts):
                 if t < curr_t + 2.9: 
                     target = t
                     break
             
        if target is not None:
             # Jump to 3 seconds before the pattern
             self.perform_seek(max(0, target - 3.0))
        else:
             msg = "No more patterns forward." if direction > 0 else "No more patterns backward."
             messagebox.showinfo("End", msg)

    def seek_backward(self):
        """Jump back 5 seconds"""
        if self.df is None: return
        # Need to handle elapsed time during playback
        curr = self.get_current_time()
        target = max(0, curr - 5.0)
        self.perform_seek(target)

    def seek_forward(self):
        """Jump forward 5 seconds"""
        if self.df is None: return
        curr = self.get_current_time()
        target = min(self.audio_len_sec, curr + 5.0)
        self.perform_seek(target)

    def get_current_time(self):
        """Helper to get current logical time regardless of playback state"""
        if self.is_playing:
            elapsed = time.time() - self.start_time
            return self.playback_offset + elapsed
        return self.playback_offset

    def animate(self):
        # [FIX] Check global app_running status if available
        try:
             import __main__
             if hasattr(__main__, 'app_running') and not __main__.app_running:
                  return
        except: pass

        if not self.is_playing: return
        elapsed = time.time() - self.start_time
        current_time = self.playback_offset + elapsed
        
        if current_time >= self.audio_len_sec:
            self.stop_playback(reset=True)
            return
            
        # Update plot
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
        
        # 1. SMART SCALING (Fitting visible window: -7s to +3s)
        med_start = current_time - WINDOW_PAST
        med_end = current_time + WINDOW_FUTURE
        idx_med_start = np.searchsorted(self.time_index, med_start)
        idx_med_end = np.searchsorted(self.time_index, med_end)
        med_view = self.df.iloc[idx_med_start:idx_med_end]
        
        if med_view.empty:
            # Fallback if no data is in view
            c_val = 1.0
            target_w = 0.125
        else:
            # Center (Pivot) is the median of the current 10s window
            c_val = med_view['GSR'].median()
            
            # [NEW] "Smart Zoom" - Calculate w_needed to fit peaks in med_view
            y_raw_m = med_view['GSR'].values
            log_vals_m = np.log10(np.maximum(0.01, y_raw_m))
            log_min_m = np.min(log_vals_m)
            log_max_m = np.max(log_vals_m)
            log_med_m = np.median(log_vals_m)
            
            # Solve for W that fits both bounds centered at med (62.5% height)
            w_for_max = (log_max_m - log_med_m) / 0.32 # 0.375 total but with margin
            w_for_min = (log_med_m - log_min_m) / 0.58 # 0.625 total but with margin
            
            needed_w = max(w_for_max, w_for_min)
            # 0.05 is a sensible (~12% span) floor to prevent over-zooming on noise
            target_w = max(0.05, needed_w)

        # [NEW] Seek-Snap: Reset smoothing if jumping > 1 second (Scrubbing/Seek)
        is_jump = abs(current_time - self.last_plot_time) > 1.0
        self.last_plot_time = current_time

        # [NEW] Asymmetric Smoothed Transitions (LERP)
        if self.smooth_c_val is None or is_jump:
            self.smooth_c_val = c_val
            self.smooth_w_val = target_w
        else:
            # Center follows at standard speed
            self.smooth_c_val = (0.10 * c_val) + (0.90 * self.smooth_c_val)
            
            # Asymmetric Window: Fast expand, very slow shrink
            if target_w > self.smooth_w_val:
                w_factor = 0.30 # Catch peaks quickly
            else:
                w_factor = 0.02 # [TUNE] Faster recovery (1-2s typical)
                
            self.smooth_w_val = (w_factor * target_w) + ((1.0 - w_factor) * self.smooth_w_val)

        eff_win = self.smooth_w_val
        log_center = math.log10(max(0.01, self.smooth_c_val))

        # [NEW] Update Span % based on smoothed window
        try:
            span_pct = (math.pow(10, eff_win) - 1.0) * 100.0
            self.txt_span.set_text(f"SPAN: {span_pct:.1f}%")
        except: pass

        # [NEW] Update TA Counter (Top Left)
        if hasattr(self, 'txt_ta_counter'):
            try:
                if idx_now < len(self.df) and 'TA Counter' in self.df.columns:
                    ta_val = self.df.iloc[idx_now]['TA Counter']
                    self.txt_ta_counter.set_text(f"TA COUNTER: {ta_val:.2f}")
                else:
                    self.txt_ta_counter.set_text("TA COUNTER: 0.0")
            except: pass
        
        # [MOD] Locked fixed 62.5% centering logic (Map data to -5 to 105)
        # log_center = math.log10(max(0.01, self.smooth_c_val))
        # eff_win = self.smooth_w_val

        if view.empty:
            self.line_gsr.set_data([], [])
        else:
            x_data = view['Rel_Time'].values - current_time 
            # Data Mapping Transformation
            y_raw = view['GSR'].values
            y_log = np.log10(np.maximum(0.01, y_raw))
            # Map log space to screen % space: 62.5 is center, eff_win is the zoom
            y_mapped = 62.5 + ((y_log - log_center) / eff_win) * 100.0
            self.line_gsr.set_data(x_data, y_mapped)

        # Update Differential TA Labels for the FIXED grid lines
        self.txt_ta_set_line.set_text(f"TA SET: {self.smooth_c_val:.2f}")
            
        unit_mults = [2, 1, -1, -2, -3]
        for i, t_label in enumerate(self.txt_grid_labels):
            if i < len(unit_mults):
                # Calculate differential TA (+/- change from center)
                target_log = log_center + (unit_mults[i] * 0.2 * eff_win)
                target_ta = math.pow(10, target_log)
                delta_ta = target_ta - self.smooth_c_val
                sign = "+" if delta_ta > 0 else ""
                t_label.set_text(f"{sign}{delta_ta:.2f} TA")


        # [NEW] Update Pattern Display from CSV
        if idx_now < len(self.df) and 'Pattern' in self.df.columns:
            pattern = self.df.iloc[idx_now]['Pattern']
            if pd.notna(pattern) and pattern != "":
                # Color logic (matching v42)
                col = 'gray'
                if pattern == "BLOWDOWN": col = '#00CED1'
                elif pattern == "ROCKET READ": col = '#DC143C'  # Crimson
                elif pattern == "LONG FALL": col = '#008000'
                elif pattern == "SHORT FALL": col = '#3CB371'
                elif pattern == "LONG RISE": col = '#FF4500'    # OrangeRed
                elif pattern == "SHORT RISE": col = 'orange'
                elif pattern == "MOTION": col = '#8B0000'       # Dark Red
                
                self.txt_pattern.set_text(f"{pattern}")
                self.txt_pattern.set_color(col)
            else:
                self.txt_pattern.set_text("")
        else:
            self.txt_pattern.set_text("")

        # 2. Minimap
        if self.mini_cursor:
            self.mini_cursor.set_xdata([current_time])

        # 3. Draw
        try:
            self.canvas.draw()
        except: pass
