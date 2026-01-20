import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, Checkbutton, BooleanVar, Scale
import sys
import os
import threading
import time
from datetime import datetime

# --- SETTINGS ---
WINDOW_SEC = 5.0
UPDATE_INTERVAL_MS = 50  # 20 FPS
DEFAULT_SMOOTHING = 6
DEFAULT_Y_SCALE = 100


class SessionReplayer:
    def __init__(self, master):
        self.master = master
        self.master.title("EEG Viewer (v23) - High Resolution")
        self.master.geometry("1400x950")

        # PROTOCOL HANDLER FOR CLEAN EXIT
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        self.df_raw = None
        self.df = None
        self.time_index = None

        self.audio_data = None
        self.audio_fs = 44100
        self.audio_len_sec = 0
        self.audio_len_sec = 0
        self.notes_content = ""
        self.notes_path = None  # [NEW]

        self.is_playing = False

        # TIMING STATE
        self.start_time = 0
        self.playback_offset = 0
        self.last_known_time = 0

        self.is_dragging = False  # Lock for seek slider
        self.bg = None

        # Band Toggles
        self.band_vars = {
            'Delta': BooleanVar(value=True),
            'Theta': BooleanVar(value=True),
            'Alpha': BooleanVar(value=True),
            'Beta': BooleanVar(value=True),
            'Gamma': BooleanVar(value=True)
        }

        # UI
        self.setup_gui()

    def setup_gui(self):
        # --- Toolbar ---
        toolbar = tk.Frame(self.master, height=50, bg='#f0f0f0')
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(toolbar, text="Load Session", command=self.load_session, bg='lightblue').pack(side=tk.LEFT, padx=5)
        self.lbl_file = tk.Label(toolbar, text="No File", bg='#f0f0f0')
        self.lbl_file.pack(side=tk.LEFT, padx=5)

        tk.Frame(toolbar, width=20, bg='#f0f0f0').pack(side=tk.LEFT)

        self.btn_play = tk.Button(toolbar, text="Play", command=self.toggle_play, state=tk.DISABLED, width=10)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        # Seek Slider
        self.slider = tk.Scale(toolbar, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0,
                               resolution=0.1, length=400)
        self.slider.pack(side=tk.LEFT, padx=5)

        # Bindings for Manual Seek
        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        self.lbl_time = tk.Label(toolbar, text="00:00 / 00:00", bg='#f0f0f0', font=('Courier', 12))
        self.lbl_time.pack(side=tk.LEFT, padx=5)

        # --- Smoothing Slider ---
        tk.Frame(toolbar, width=20, bg='#f0f0f0').pack(side=tk.LEFT)
        tk.Label(toolbar, text="Smoothing:", bg='#f0f0f0').pack(side=tk.LEFT)
        self.scale_smooth = tk.Scale(toolbar, from_=1, to=50, orient=tk.HORIZONTAL, length=100)
        self.scale_smooth.set(DEFAULT_SMOOTHING)
        self.scale_smooth.pack(side=tk.LEFT, padx=5)
        self.scale_smooth.bind("<ButtonRelease-1>", self.on_smooth_change)

        # --- Band Toggles ---
        toggle_frame = tk.Frame(self.master, bg='#e0e0e0', bd=1, relief=tk.SUNKEN)
        toggle_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        tk.Label(toggle_frame, text="Bands:", bg='#e0e0e0', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)

        colors = {'Delta': 'navy', 'Theta': 'teal', 'Alpha': 'orange', 'Beta': 'red', 'Gamma': 'purple'}
        for b in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
            cb = Checkbutton(toggle_frame, text=b, variable=self.band_vars[b],
                             bg='#e0e0e0', fg=colors[b], selectcolor='white',
                             command=self.refresh_visibility)
            cb.pack(side=tk.LEFT, padx=5)

        # --- Main Area ---
        panes = tk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        # Left: Graph Container
        self.graph_container = tk.Frame(panes, bg='white')
        panes.add(self.graph_container, minsize=900)

        # [NEW] Vertical Scale Slider (Y-Axis Zoom)
        # We put this on the LEFT of the graph
        self.scale_y = tk.Scale(self.graph_container, from_=100, to=10, orient=tk.VERTICAL,
                                length=500, label="Scale")
        self.scale_y.set(DEFAULT_Y_SCALE)
        self.scale_y.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        # Bind to live update? Or release? Live update might be heavy but let's try release first for safety.
        # Actually with Blitting, live update is risky if we change limits (requires full redraw).
        # We'll use Release.
        self.scale_y.bind("<ButtonRelease-1>", self.on_yscale_change)

        # The Matplotlib Canvas Area
        self.graph_frame = tk.Frame(self.graph_container, bg='white')
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.graph_frame.bind("<Configure>", self.on_resize)

        # Right: Info/Notes
        self.info_frame = tk.Frame(panes, width=300, bg='#fafafa')
        panes.add(self.info_frame, minsize=300)

        # Notes
        tk.Label(self.info_frame, text="Session Notes", font=('Arial', 12, 'bold')).pack(pady=10)
        self.txt_notes = scrolledtext.ScrolledText(self.info_frame, height=10, wrap=tk.WORD)
        self.txt_notes.pack(fill=tk.X, padx=10)

        # [NEW] Save Notes Button (Moved Up)
        self.btn_save_notes = tk.Button(self.info_frame, text="Save Notes", command=self.save_notes, bg='#ddffdd')
        self.btn_save_notes.pack(fill=tk.X, padx=10, pady=5)

        # Stats
        tk.Label(self.info_frame, text="Statistics", font=('Arial', 12, 'bold')).pack(pady=5)
        self.lbl_stats = tk.Label(self.info_frame, text="--", justify=tk.LEFT)
        self.lbl_stats.pack(fill=tk.X, padx=10)

        # Debug Log
        tk.Label(self.info_frame, text="Debug Log", font=('Arial', 10, 'bold')).pack(pady=5)
        self.txt_debug = scrolledtext.ScrolledText(self.info_frame, height=10, state='disabled')
        self.txt_debug.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- Matplotlib ---
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.init_plot()

    def on_close(self):
        self.stop_playback(reset=False)
        plt.close(self.fig)
        self.master.destroy()
        sys.exit(0)

    def log(self, msg):
        try:
            self.txt_debug.config(state='normal')
            self.txt_debug.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")
            self.txt_debug.see(tk.END)
            self.txt_debug.config(state='disabled')
            print(msg)
        except:
            pass

    def save_notes(self):
        if not self.notes_path:
            self.log("No notes file loaded to save to.")
            return
        try:
            content = self.txt_notes.get("1.0", tk.END).strip()
            with open(self.notes_path, "w") as f:
                f.write(content)
            self.log(f"Notes Saved: {os.path.basename(self.notes_path)}")
            messagebox.showinfo("Saved", "Notes updated successfully.")
        except Exception as e:
            self.log(f"Save Error: {e}")
            messagebox.showerror("Error", f"Failed to save notes: {e}")

    def on_yscale_change(self, event):
        # Re-init plot to set new limits and capture new background
        self.init_plot()
        # Draw current data on top
        self.update_plot(self.playback_offset, force_draw=True)

    def init_plot(self):
        self.ax.clear()
        self.ax.set_title("Brainwave Band Power")

        # [NEW] Dynamic Y-Scale
        y_max = self.scale_y.get()
        self.ax.set_ylim(0, y_max)

        self.ax.set_xlim(-WINDOW_SEC, 0)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylabel("Power")
        self.ax.set_xlabel("Time (s relative to Playhead)")

        self.lines = {}
        colors = {'Delta': 'navy', 'Theta': 'teal', 'Alpha': 'orange', 'Beta': 'red', 'Gamma': 'purple'}

        for b in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
            line, = self.ax.plot([], [], label=b, color=colors[b], lw=2)
            self.lines[b] = line

        self.ax.legend(loc='upper left')
        try:
            self.canvas.draw()
            # Capture background for blitting
            self.bg = self.canvas.copy_from_bbox(self.ax.bbox)
        except:
            pass

    def on_resize(self, event):
        self.init_plot()

    def refresh_visibility(self):
        for b, line in self.lines.items():
            line.set_visible(self.band_vars[b].get())
        if not self.is_playing:
            self.update_plot(self.playback_offset, force_draw=True)

    def load_session(self):
        path = filedialog.askopenfilename(initialdir="Session_Data",
                                          filetypes=[("CSV Files", "eeg.csv"), ("All CSV", "*.csv")])
        if not path: return

        try:
            self.lbl_file.config(text=os.path.basename(path))
            self.log(f"Loading: {os.path.basename(path)}")

            self.df_raw = pd.read_csv(path)
            self.df_raw.columns = [c.strip() for c in self.df_raw.columns]
            
            # [FIX] v21 Column Compatibility
            # Map EEG_Delta -> Delta etc
            rename_map = {
                'EEG_Delta': 'Delta', 'EEG_Theta': 'Theta', 'EEG_Alpha': 'Alpha',
                'EEG_Beta': 'Beta', 'EEG_Gamma': 'Gamma',
                'Elapsed_Sec': 'Rel_Time', 
                'Delta_1': 'Delta', 'Theta_1': 'Theta' # Fallback for other versions
            }
            self.df_raw.rename(columns=rename_map, inplace=True)

            # SORT TIME
            if 'Rel_Time' in self.df_raw.columns:
                self.df_raw.sort_values(by='Rel_Time', inplace=True)
            elif 'Timestamp' in self.df_raw.columns and not 'Rel_Time' in self.df_raw.columns:
                 # Fallback calc
                 start_t = self.df_raw['Timestamp'].iloc[0]
                 self.df_raw['Rel_Time'] = self.df_raw['Timestamp'] - start_t

            self.apply_smoothing()

            # Audio Load
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            
            # [FIX] v21 Audio Path (audio.wav in same folder)
            audio_path = os.path.join(dirname, "audio.wav")
            if not os.path.exists(audio_path):
                 # Fallback to v10 logic
                parts = basename.replace(".csv", "").split("_")
                if len(parts) >= 2:
                    ts = f"{parts[-2]}_{parts[-1]}"
                else:
                    ts = "UNKNOWN"
                audio_path = os.path.join(dirname, f"brainwave_audio_{ts}.wav")

            # [FIX] v21 Notes Path
            notes_path = os.path.join(dirname, "notes.txt")
            if not os.path.exists(notes_path):
                notes_path = os.path.join(dirname, f"brainwave_notes_{ts}.txt") if 'ts' in locals() else ""
            
            self.notes_path = notes_path  # [NEW] Store ref

            if os.path.exists(audio_path):
                data, fs = sf.read(audio_path, dtype='float32')
                self.audio_data = data
                self.audio_fs = fs
                # Fix: len(data) might be shape (N, 2)
                if len(self.audio_data.shape) > 1:
                     # Flatten or just take length of first dim
                     self.audio_len_sec = self.audio_data.shape[0] / self.audio_fs
                else:
                     self.audio_len_sec = len(self.audio_data) / self.audio_fs
                
                self.log(f"Audio Loaded: {self.audio_len_sec:.2f}s")
            else:
                self.audio_data = np.zeros((44100 * 10), dtype='float32')
                self.audio_fs = 44100
                self.audio_len_sec = 10
                if self.df is not None and 'Rel_Time' in self.df.columns:
                    self.audio_len_sec = self.df['Rel_Time'].max()
                self.log("No audio found (Silent Mode)")

            if self.notes_path and os.path.exists(self.notes_path):
                with open(self.notes_path, 'r') as f:
                    self.notes_content = f.read()
            else:
                self.notes_content = "No notes."

            self.txt_notes.delete('1.0', tk.END)
            self.txt_notes.insert(tk.END, self.notes_content)

            self.lbl_stats.config(text=f"Dur: {self.audio_len_sec:.1f}s")

            # Reset
            self.is_dragging = True  # Prevention
            self.slider.config(to=self.audio_len_sec)
            self.slider.set(0)
            self.is_dragging = False  # Release

            self.playback_offset = 0
            self.stop_playback(reset=True)
            self.btn_play.config(state=tk.NORMAL)
            self.init_plot()
            self.update_plot(0, force_draw=True)

        except Exception as e:
            self.log(f"Load Crash: {e}")
            import traceback;
            traceback.print_exc()

    def on_smooth_change(self, event):
        if self.df_raw is None: return
        self.apply_smoothing()
        self.update_plot(self.playback_offset, force_draw=True)

    def apply_smoothing(self):
        self.df = self.df_raw.copy()
        win = self.scale_smooth.get()
        if win > 1:
            for b in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                if b in self.df.columns:
                    self.df[b] = self.df[b].rolling(window=win, min_periods=1).mean()

        # PREPARE FOR FAST BINARY SEARCH (Optimize CPU)
        if 'Rel_Time' in self.df.columns:
            self.time_index = self.df['Rel_Time'].values
        else:
            self.time_index = None

    def toggle_play(self):
        if self.is_playing:
            # Just pause (stop audio, keep offset)
            # Calculate where we are exact
            elapsed = time.time() - self.start_time
            self.playback_offset += elapsed
            self.stop_playback(reset=False)
        else:
            self.start_playback()

    def start_playback(self):
        if self.is_playing: return
        try:
            self.is_playing = True
            self.btn_play.config(text="Pause")

            # AUDIO
            start_frame = int(self.playback_offset * self.audio_fs)
            if start_frame < len(self.audio_data):
                sd.play(self.audio_data[start_frame:], self.audio_fs)
                self.start_time = time.time()
                self.log(f"Playing from {self.playback_offset:.1f}s")
                self.animate()
            else:
                self.stop_playback(reset=True)

        except Exception as e:
            self.log(f"Start Error: {e}")
            self.stop_playback()

    def stop_playback(self, reset=True):
        self.is_playing = False
        self.btn_play.config(text="Play")

        try:
            sd.stop()
        except:
            pass

        if reset:
            self.playback_offset = 0
            self.is_dragging = True
            self.slider.set(0)
            self.is_dragging = False
            self.init_plot()
            self.update_plot(0, force_draw=True)

    def on_slider_press(self, event):
        self.is_dragging = True

    def on_slider_release(self, event):
        self.is_dragging = False
        # Now we seek
        val = self.slider.get()
        self.perform_seek(val)

    def perform_seek(self, val):
        was_playing = self.is_playing
        if was_playing:
            self.stop_playback(reset=False)

        ts = float(val)
        self.playback_offset = ts

        self.update_plot(ts, force_draw=True)
        self.log(f"Seek to {ts:.1f}s")

        if was_playing:
            self.start_playback()

    def animate(self):
        if not self.is_playing: return

        # SYSTEM CLOCK DRIVE
        elapsed = time.time() - self.start_time
        current_time = self.playback_offset + elapsed

        if current_time >= self.audio_len_sec:
            self.stop_playback(reset=True)
            return

        # Only update slider if user is NOT dragging it
        if not self.is_dragging:
            self.slider.set(current_time)

        self.update_plot(current_time)
        self.master.after(UPDATE_INTERVAL_MS, self.animate)

    def update_plot(self, current_time, force_draw=False):
        t_str = f"{int(current_time) // 60:02}:{int(current_time) % 60:02}"
        tot_str = f"{int(self.audio_len_sec) // 60:02}:{int(self.audio_len_sec) % 60:02}"
        self.lbl_time.config(text=f"{t_str} / {tot_str}")

        if self.df is None or self.time_index is None: return

        # PERFORMANCE OPTIMIZATION: BINARY SEARCH SLICING
        win_start = current_time - WINDOW_SEC
        idx_start = np.searchsorted(self.time_index, win_start, side='left')
        idx_end = np.searchsorted(self.time_index, current_time, side='right')

        view = self.df.iloc[idx_start:idx_end]

        x_data = []
        if not view.empty:
            x_data = view['Rel_Time'].values - current_time

        # Update Data
        for b, line in self.lines.items():
            if line.get_visible():
                if view.empty:
                    line.set_data([], [])
                else:
                    line.set_data(x_data, view[b].values)

        # ARTIFACT-FREE BLIT
        if self.bg and not force_draw:
            try:
                self.canvas.restore_region(self.bg)
                for b, line in self.lines.items():
                    if line.get_visible(): self.ax.draw_artist(line)
                self.canvas.blit(self.ax.bbox)
            except:
                self.canvas.draw()
        else:
            self.canvas.draw()


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SessionReplayer(root)
        root.mainloop()
    except Exception as e:
        print(f"Crash: {e}")
