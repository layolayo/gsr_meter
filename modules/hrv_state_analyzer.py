import numpy as np
from collections import deque

class HRVStateAnalyzer:
    def __init__(self, window_size=300):
        """
        Analyzes HR/HRV data to determine Psychophysiological State and Trend.
        
        Args:
            window_size (int): Number of samples to keep for baseline calculation. 
                               Default 300 samples (at 1Hz = 5 mins, at 4Hz = ~75s).
        """
        self.hr_history = deque(maxlen=window_size)
        self.hrv_history = deque(maxlen=window_size)
        self.prev_quadrant = None
        self.current_quadrant = None
        
        # [NEW] Sticky Trend Logic
        self.active_trend = "Sustained"
        self.trend_start_time = 0
        self.TREND_HOLD_SEC = 3.0 # Duration to hold trend text
        
        # State Definitions
        self.QUADRANTS = {
            1: "Stress",      # HR High, HRV Low (Top-Left)
            2: "Flow",        # HR High, HRV High (Top-Right)
            3: "Recovery",    # HR Low, HRV High (Bottom-Right)
            4: "Withdrawal"   # HR Low, HRV Low (Bottom-Left)
        }
        
        self.TRANSITIONS = {
            # From Stress (1)
            (1, 2): "Reappraisal",
            (1, 3): "Reset",
            (1, 4): "Collapse",
            # From Flow (2)
            (2, 1): "Overwhelm",
            (2, 3): "Deactivation",
            (2, 4): "Crash",
            # From Recovery (3)
            (3, 1): "Alarm",
            (3, 2): "Engagement",
            (3, 4): "Depletion",
            # From Withdrawal (4)
            (4, 1): "Agitation",
            (4, 2): "Vitality",
            (4, 3): "Restoration"
        }

    def update(self, hr, hrv):
        """
        Updates the analyzer with new sensor data.
        
        Args:
            hr (float): Heart Rate in BPM
            hrv (float): Heart Rate Variability (RMSSD) in ms
            
        Returns:
            dict: {
                'state': str (e.g., "Stress"),
                'intensity': str (e.g., "High"),
                'trend': str (e.g., "Reappraisal"),
                'quadrant': int (1-4),
                'z_hr': float,
                'z_hrv': float
            }
        """
        # Validate inputs
        if hr is None or hrv is None or hr == 0:
            return self._empty_result()

        # Update History
        self.hr_history.append(hr)
        self.hrv_history.append(hrv)
        
        # Need enough data for baseline? 
        # For now, if < 10 samples, just return "Calibrating" or raw quadrant vs static mean?
        # Let's use a dynamic mean/std if we have > 10 samples, otherwise use generic population norms?
        # Converting to Z-scores requires a baseline. 
        # Let's use the running history as the "baseline" (self-referential).
        
        if len(self.hr_history) < 10:
             return self._empty_result(status="Calibrating")

        # Calculate Z-Scores based on recent history (Self-Referential)
        hr_mean = np.mean(self.hr_history)
        hr_std = np.std(self.hr_history)
        hrv_mean = np.mean(self.hrv_history)
        hrv_std = np.std(self.hrv_history)
        
        # Enforce minimum deviation to prevent noise amplification
        # [Adjusted] HR: 3.0 BPM, HRV: 5.0 ms (Lowered from 8.0 to restore movement)
        hr_std = max(hr_std, 3.0)
        hrv_std = max(hrv_std, 5.0)
        
        raw_z_hr = (hr - hr_mean) / hr_std
        raw_z_hrv = (hrv - hrv_mean) / hrv_std
        
        # [NEW] Smooth Z-Scores (EMA)
        # alpha = 0.08 (Medium-Slow), was 0.05 (Too Slow)
        alpha_hr = 0.1   # HR is smoother naturally
        alpha_hrv = 0.08 # Increased alpha to restore responsiveness
        
        if not hasattr(self, 'smooth_z_hr'): self.smooth_z_hr = 0.0
        if not hasattr(self, 'smooth_z_hrv'): self.smooth_z_hrv = 0.0
        
        self.smooth_z_hr = alpha_hr * raw_z_hr + (1 - alpha_hr) * self.smooth_z_hr
        self.smooth_z_hrv = alpha_hrv * raw_z_hrv + (1 - alpha_hrv) * self.smooth_z_hrv
        
        z_hr = self.smooth_z_hr
        z_hrv = self.smooth_z_hrv
        
        # Determine Quadrant
        # Axis 1 (Y): HR -> High (>0) vs Low (<0)
        # Axis 2 (X): HRV -> High (>0) vs Low (<0)
        
        # Top-Left (1): HR High, HRV Low -> z_hr > 0, z_hrv < 0
        # Top-Right (2): HR High, HRV High -> z_hr > 0, z_hrv > 0
        # Bottom-Right (3): HR Low, HRV High -> z_hr < 0, z_hrv > 0
        # Bottom-Left (4): HR Low, HRV Low -> z_hr < 0, z_hrv < 0
        
        quad = 0
        if z_hr >= 0 and z_hrv < 0:
            quad = 1 # Stress
        elif z_hr >= 0 and z_hrv >= 0:
            quad = 2 # Flow
        elif z_hr < 0 and z_hrv >= 0:
            quad = 3 # Recovery
        else:
            quad = 4 # Withdrawal
            
        # Determine Intensity (Max Z-score magnitude)
        max_z = max(abs(z_hr), abs(z_hrv))
        intensity = "Low"
        if max_z > 2.0:
            intensity = "High"
        elif max_z > 1.0:
            intensity = "Medium"
            
        # Determine Trend
        # [NEW] Sticky Trend Implementation
        import time
        now = time.time()
        
        new_trend = None
        
        if self.current_quadrant and self.current_quadrant != quad:
            # Transition occurred
            key = (self.current_quadrant, quad)
            new_trend = self.TRANSITIONS.get(key, "Transition")
            self.prev_quadrant = self.current_quadrant
            
            # Update Active Trend
            self.active_trend = new_trend
            self.trend_start_time = now
            
        # Check Expiry
        if now - self.trend_start_time > self.TREND_HOLD_SEC:
            self.active_trend = "Sustained"
            
        trend = self.active_trend
            
        self.current_quadrant = quad
        
        state_name = self.QUADRANTS[quad]
        
        return {
            'state': state_name,
            'intensity': intensity,
            'trend': trend,
            'quadrant': quad,
            'z_hr': f"{z_hr:.3f}", # String for CSV friendly
            'z_hrv': f"{z_hrv:.3f}",
            'status': "Active"
        }

    def _empty_result(self, status="Waiting"):
        return {
            'state': "Unknown",
            'intensity': "-",
            'trend': "-",
            'quadrant': 0,
            'z_hr': 0.0,
            'z_hrv': 0.0,
            'status': status
        }
