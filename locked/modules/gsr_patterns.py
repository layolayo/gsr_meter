
import collections
import numpy as np
import time

class GSRPatterns:
    def __init__(self, history_len_sec=3.0, sample_rate=20.0):
        self.history_len_sec = history_len_sec
        self.fs = sample_rate
        self.maxlen = int(history_len_sec * sample_rate)
        
        # Buffer: list of (timestamp, value)
        self.buffer = collections.deque(maxlen=self.maxlen)
        self.last_update_time = 0
        
        # State
        self.current_pattern = ""
        self.pattern_confidence = 0.0

        self.DIAL_SIZE_INCHES = 5.0
        
        self.THRESH_FALL_LARGE = 1.5   # "Drop 1/3 of a dial"
        self.THRESH_STUCK = 0.125      # [MOD] Reverted to 0.125 (per user req, with 2s duration)
       
    def update(self, ta_val, timestamp=None, effective_window=5.0, is_motion=False):
        if timestamp is None: timestamp = time.time()
        
        # [NEW] Motion Block
        if is_motion:
            return "MOTION"
        
        self.buffer.append((timestamp, ta_val))
        if len(self.buffer) < self.fs * 1.0: # Need at least 1s
            return ""
            
        # Extract arrays
        vals_ta = np.array([x[1] for x in self.buffer])
        times = np.array([x[0] for x in self.buffer])
        
        # --- CONVERT TO INCHES (LOG SPACE) ---
        # Inches = (Log10(TA) / LogWindow) * 5.0
        # This makes pattern detection match visual movement on the Log Graph.
        log_vals = np.log10(np.maximum(0.01, vals_ta))
        scale_factor = (self.DIAL_SIZE_INCHES / effective_window) # effective_window is LOG_WINDOW_HEIGHT
        vals_inches = log_vals * scale_factor
        
        # --- CALCULATE METRICS (IN INCHES) ---
        
        # 1. Trend (Last 3s)
        # [FIX] Explictly use 3.0s window instead of full buffer (8.0s)
        idx_trend = max(0, len(vals_inches) - int(self.fs * 3.0))
        net_change = vals_inches[-1] - vals_inches[idx_trend]
        duration = times[-1] - times[idx_trend]
        if duration < 0.1: duration = 0.1

            
        # --- PATTERN LOGIC ---
        
        # Priority 1: High Energy / Blowdown / Rocket
        # "Fast Fall" or "Fast Rise"
        # [MOD] Added raw TA check to prevent High-Sens noise 
        # User: "TA change is not 0.2" -> Enforce 0.2 minimum RAW TA change.
        net_change_ta = vals_ta[-1] - vals_ta[0]
        
        # 2. Instant Velocity (Last 0.2s)
        idx_inst = max(0, len(vals_inches) - 5)
        inst_change = vals_inches[-1] - vals_inches[idx_inst]
        inst_dur = times[-1] - times[idx_inst]
        if inst_dur < 0.01: inst_dur = 0.01
        inst_vel = abs(inst_change / inst_dur)
        
        if abs(inst_vel) > 2.0:
             if net_change < -self.THRESH_FALL_LARGE and abs(net_change_ta) >= 0.2:
                  return "BLOWDOWN"
             elif net_change > self.THRESH_FALL_LARGE and abs(net_change_ta) >= 0.2:
                  return "ROCKET READ"

        # Priority 2: Stuck Needle (Updated: 3.0s, Tight)
        # User: "Time wasn't long enough / Sensitivity too much" -> Harder to trigger.
        #idx_stuck = max(0, len(vals_inches) - int(self.fs * 3.0))
        #var_win_stuck = vals_inches[idx_stuck:]
        #if len(var_win_stuck) > self.fs * 2.5: # Ensure we have enough data
        #     amp_stuck = np.max(var_win_stuck) - np.min(var_win_stuck)
        #     # [MOD] Tightened to 0.02 inches (Literally no movement)
        #     if amp_stuck < 0.02: 
        #          return "" # Was STUCK

        # Priority 3: Trends vs Reads (Velocity Dependent)
        # User: "Steady falls are just falls... sudden are short/long"
        
        # Calculate Trend Velocity (Inches / Sec for the window)
        trend_vel = abs(net_change / duration)
        
        # Case A: Falling
        if net_change < -self.THRESH_STUCK:
             mag = abs(net_change)
             
             # If "Sudden" (Velocity > 0.15 in/sec) -> It's a Read (Short/Long)
             if trend_vel > 0.15:
                # [MOD] Filter tiny fast drops (Noise/Glitches < 0.2)
                if mag < 0.2: return "TICK" 
                elif mag < self.THRESH_FALL_LARGE: return "SHORT FALL"
                else: return "LONG FALL"
             else:
                 return "FALL" # Steady Drop
        
        # Case B: Rising
        if net_change > self.THRESH_STUCK:
             mag = abs(net_change)
             
             if trend_vel > 0.15:
                 if mag < 0.2: return "RISE"
                 elif mag < self.THRESH_FALL_LARGE: return "SHORT RISE"
                 else: return "LONG RISE"
             else:
                 return "RISE" # Steady Rise
                 
        return ""
