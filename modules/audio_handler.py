import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tkinter as tk
from tkinter import ttk
import time

import queue
import wave
import threading
import os

class AudioHandler:
    def __init__(self, log_callback, ui_callback):
        """
        :param log_callback: func(msg: str)
        :param ui_callback: func(key: str, val: Any) -> Handles UI updates
                            Keys: 'mic_name_text', 'status_text'
        """
        self.log_cb = log_callback
        self.ui_cb = ui_callback
        
        # State
        self.audio_queue = queue.Queue()
        self.audio_state = {'peak': 0}
        self.current_mic_name = "Default"
        self.selected_device_idx = None
        self.current_mic_gain = 3.0
        self.current_mic_rate = None
        self.is_recording = False
        self.audio_stream = None
        
        # Writer State
        self.writer_thread = None
        self.writer_running = False
        
    def log(self, msg):
        if self.log_cb: self.log_cb(msg)
        else: print(msg)

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice InputStream"""
        gained_sig = indata * self.current_mic_gain
        peak = np.max(np.abs(gained_sig))
        self.audio_state['peak'] = peak 
        
        if self.is_recording:
             amplified = np.clip(gained_sig, -1.0, 1.0)
             # Push to queue for background writing
             try:
                 self.audio_queue.put_nowait(amplified.copy())
             except: pass

    def start_recording(self, filename):
        """Start Streaming Recording"""
        if self.is_recording: return
        
        self.is_recording = True
        self.writer_running = True
        
        # Determine Rate
        fs = self.current_mic_rate if self.current_mic_rate else 44100
        
        # Start Thread
        self.writer_thread = threading.Thread(target=self._writer_loop, args=(filename, fs), daemon=True)
        self.writer_thread.start()
        self.log(f"Audio Recording Started: {filename} ({fs}Hz)")

    def stop_recording(self):
        """Stop Streaming Recording"""
        if not self.is_recording: return
        
        self.is_recording = False
        self.writer_running = False
        
        if self.writer_thread:
            self.writer_thread.join(timeout=2.0)
            self.writer_thread = None
            
        self.log("Audio Recording Stopped")

    def _writer_loop(self, filename, fs):
        """Background Thread to write WAV file"""
        try:
            # 16-bit PCM WAV
            # [FIX] Open file explicitly to allow fsync
            with open(filename, 'wb') as f:
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(fs)
                    
                    last_flush = time.time()
                    last_log = time.time()
                    
                    while self.writer_running or not self.audio_queue.empty():
                        try:
                            # Blocking get with timeout to allow checking running flag
                            chunk = self.audio_queue.get(timeout=0.5)
                            
                            # Convert float32 [-1, 1] to int16
                            # Ensure nice clipping
                            chunk_int16 = (chunk * 32767).astype(np.int16)
                            wf.writeframes(chunk_int16.tobytes())
                            
                            # [NEW] Periodic Safe Flush (Every ~1.0s)
                            now = time.time()
                            if now - last_flush > 1.0:
                                 try:
                                     # Force OS write to prevent buffer build-up (CPU Spike)
                                     f.flush()
                                     os.fsync(f.fileno())
                                     last_flush = now
                                 except Exception as e: pass

                            # [NEW] Debug Queue Size (Every 5s)
                            if now - last_log > 5.0:
                                 q_sz = self.audio_queue.qsize()
                                 if q_sz > 5:
                                     self.log(f"Audio Queue Size: {q_sz}")
                                 last_log = now
                            
                        except queue.Empty:
                            continue
                        except Exception as e:
                            self.log(f"Audio Write Error: {e}")
                            break
        except Exception as e:
            self.log(f"WAV File Error: {e}") 

    def sync_audio_stream(self, target_view='main'):
        """Manages Stream Lifecycle"""
        should_be_on = self.is_recording or (target_view == 'settings')

        if self.selected_device_idx is None:
            # ... (omitted for brevity, assume unchanged logic here) ...
            pass # (This is just context, actual replacement handles the blocksize below)

    # I need to target the blocksize change specifically in sync_audio_stream, 
    # but replace_file_content works best with contiguous blocks. 
    # Let me split this into two replacements or target the specific areas.
    # The user instruction was "Revert blocksize to 1024".

  

    def sync_audio_stream(self, target_view='main'):
        """Manages Stream Lifecycle"""
        should_be_on = self.is_recording or (target_view == 'settings')

        if self.selected_device_idx is None:
            try:
                devs = sd.query_devices()
                target_mic = self.current_mic_name
                self.selected_device_idx = None
                self.current_mic_name = "NO MIC" 
                
                # Probe Target (Two-Pass System)
                # Pass 1: Exact Match (High Priority)
                if target_mic and target_mic != "Default" and target_mic != "N/A":
                     for i, d in enumerate(devs):
                         if d['max_input_channels'] > 0 and d['name'] == target_mic:
                              try:
                                  with sd.InputStream(device=i, samplerate=None, channels=1) as s:
                                      s.read(10)
                                  self.selected_device_idx = i
                                  self.current_mic_name = d['name'] 
                                  self.log(f"Mic Verified (Exact): {self.current_mic_name}")
                                  break
                              except Exception as pe:
                                  self.log(f"Probe Fail (Exact): {d['name']} ({pe})")

                # Pass 2: Fuzzy Match (Fallback)
                if self.selected_device_idx is None and target_mic and target_mic != "Default" and target_mic != "N/A":
                     for i, d in enumerate(devs):
                         if d['max_input_channels'] > 0 and (target_mic in d['name'] or d['name'] in target_mic):
                              try:
                                  with sd.InputStream(device=i, samplerate=None, channels=1) as s:
                                      s.read(10)
                                  self.selected_device_idx = i
                                  self.current_mic_name = d['name'] 
                                  self.log(f"Mic Verified (Fuzzy): {self.current_mic_name}")
                                  break
                              except Exception as pe:
                                  self.log(f"Probe Fail (Fuzzy): {d['name']} ({pe})")
                
                # Probe Fallback (Any working)
                if self.selected_device_idx is None:
                    for i, d in enumerate(devs):
                        if d['max_input_channels'] > 0:
                             try:
                                 with sd.InputStream(device=i, samplerate=None, channels=1) as s:
                                     s.read(10)
                                 self.selected_device_idx = i
                                 self.current_mic_name = d['name']
                                 self.log(f"Mic Verified (Fallback): {self.current_mic_name}")
                                 break
                             except: pass

                # Update UI
                self.ui_cb('mic_name_text', self.current_mic_name)
                # Parse shorter name for status bar
                short_name = self.current_mic_name
                if "(" in short_name: short_name = short_name.split("(")[0].strip()
                if len(short_name) > 20: short_name = short_name[:20] + "..."
                self.ui_cb('status_text', f"AUDIO: {short_name}")
                     
            except Exception as e: self.log(f"Dev Check Err: {e}")

        if not should_be_on:
            if self.audio_stream:
                self.audio_stream.stop(); self.audio_stream.close(); self.audio_stream = None
                self.log("Audio Stream: OFF")
            return

        if self.audio_stream and self.audio_stream.active:
             return 
             
        try:
            if self.selected_device_idx is None: return 
            
            # Rate Selection
            rates_to_try = []
            if self.current_mic_rate: rates_to_try.append(self.current_mic_rate)
            rates_to_try.extend([44100, 48000, None])
            rates_to_try = list(dict.fromkeys(rates_to_try))
            
            stream_created = False
            
            if self.audio_stream:
                 try: self.audio_stream.close()
                 except: pass
                 self.audio_stream = None
            time.sleep(0.2) 
            
            for sr in rates_to_try:
                try:
                    self.log(f"Trying SR: {sr if sr else 'Auto'}...")
                    self.audio_stream = sd.InputStream(
                        samplerate=sr, channels=1, device=self.selected_device_idx, 
                        callback=self.audio_callback, blocksize=1024
                    )
                    self.audio_stream.start()
                    stream_created = True
                    actual_rate = sr if sr else self.audio_stream.samplerate
                    self.log(f"Audio Stream: ON ({actual_rate} Hz)")
                    
                    self.current_mic_rate = int(actual_rate)
                    break
                except Exception as e:
                    self.log(f"SR {sr} Fail: {e}")
                    if self.audio_stream: 
                        try: self.audio_stream.close()
                        except: pass; 
                        self.audio_stream = None
                    time.sleep(0.3) 
            
            if not stream_created:
                raise Exception("All Sample Rates Failed")

        except Exception as e: self.log(f"Stream Err: {e}")

    def open_audio_select(self):
         """Displays Tkinter Dialog for Mic Selection"""
         root = tk.Tk()
         root.withdraw()
         devices = sd.query_devices()
         input_devices = []
         for i, d in enumerate(devices):
             if d['max_input_channels'] > 0:
                 # Include HostAPI index or extra info to differentiate
                 extra = f" [In:{d['max_input_channels']}]"
                 input_devices.append(f"{i}: {d['name']}{extra}")
         
         if not input_devices:
             self.log("No Input Devices Found!")
             root.destroy(); return
         
         dlg = tk.Toplevel(root)
         dlg.title("Select Microphone")
         dlg.geometry("400x150")
         tk.Label(dlg, text="Choose Input Device:").pack(pady=5)
         combo = ttk.Combobox(dlg, values=input_devices, width=50)
         
         idx_to_sel = 0
         if self.selected_device_idx is not None:
             for j, d_str in enumerate(input_devices):
                 if d_str.startswith(f"{self.selected_device_idx}:"):
                     idx_to_sel = j; break
         combo.current(idx_to_sel)
         combo.pack(pady=5)
         
         user_choice = {"idx": None, "name": None}
         def on_ok():
             selection = combo.get()
             if selection:
                 idx = int(selection.split(":", 1)[0])
                 user_choice["idx"] = idx
                 # Extract name part only (remove "0: " prefix and extra info if needed, but keeping full string is safer for matching)
                 # Actually, d['name'] from query_devices is what we want.
                 # Let's re-fetch it using index to be safe/clean
                 raw_name = selection.split(":", 1)[1].strip()
                 # Remove our appended extra info?
                 if " [" in raw_name: raw_name = raw_name.rsplit(" [", 1)[0]
                 user_choice["name"] = raw_name
             dlg.destroy()
         tk.Button(dlg, text="OK", command=on_ok).pack(pady=10)
         root.wait_window(dlg)
         
         if user_choice["idx"] is not None:
             self.selected_device_idx = user_choice["idx"]
             self.current_mic_name = user_choice["name"]
             self.log(f"Selected: {self.current_mic_name}")
             
             self.ui_cb('mic_name_text', self.current_mic_name)
             
             if self.audio_stream:
                  self.audio_stream.stop(); self.audio_stream.close(); self.audio_stream = None
             
             # Re-sync immediately
             self.sync_audio_stream(target_view='settings') 

         root.destroy()
