import threading
import struct
import collections
import math
import time
from openant.easy.node import Node
from openant.easy.channel import Channel
from openant.devices import ANTPLUS_NETWORK_KEY


class AntHrvSensor:
    def __init__(self):
        self.running = False
        self.status = "Initializing"

        # --- HR DATA (Channel 0) ---
        self.bpm = 0
        self.rr_ms = 0
        self.raw_rr_ms = 0
        self.rmssd = 0.0
        self.last_raw_hex = ""

        # --- METADATA ---
        self.manufacturer_id = None
        self.serial_number = None
        self.battery_voltage = None
        self.battery_status = "Unknown"
        self.operating_time_hours = 0.0

        # --- INTERNAL BUFFERS ---
        self.rr_buffer = collections.deque(maxlen=30)
        self.filter_buffer = collections.deque(maxlen=5)
        self.consecutive_rejections = 0

        self.node = None
        self.channel_hr = None
        self.channel_run = None
        self.thread = None

        self.last_beat_time = None
        self.last_beat_count = None
        self.last_hr_data_time = time.time()

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.node:
            try:
                self.node.stop()
            except:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def get_data(self):
        """Returns merged data from both channels"""
        # Safety check on HR stream timeout
        if (time.time() - self.last_hr_data_time) > 4.0 and self.status == "Active":
            self.status = "Signal Lost"
            self.bpm = 0

        # Manufacturer Name
        manuf = "Unknown"
        if self.manufacturer_id == 1:
            manuf = "Garmin"
        elif self.manufacturer_id == 123:
            manuf = "Polar"
        elif self.manufacturer_id == 33:
            manuf = "Wahoo"
        elif self.manufacturer_id:
            manuf = f"ID {self.manufacturer_id}"

        return {
            'bpm': self.bpm,
            'rmssd': self.rmssd,
            'rr_ms': self.rr_ms,
            'raw_rr_ms': self.raw_rr_ms,
            'raw_hex': self.last_raw_hex,
            'status': self.status,
            'manufacturer': manuf,
            'serial': self.serial_number,
            'battery_volts': self.battery_voltage,
            'battery_state': self.battery_status,
            'uptime_hours': self.operating_time_hours
        }

    # --- CHANNEL 0: HEART RATE MONITOR ---
    def _on_hr_data(self, data):
        self.last_hr_data_time = time.time()
        # Capture Raw Hex for Debugging/CSV
        try:
             self.last_raw_hex = "".join([f"{x:02X}" for x in data])
        except: self.last_raw_hex = ""
        
        page = data[0] & 0x7F

        # 1. Parse Metadata Pages
        if page == 7:  # Battery
            coarse = data[3] & 0x0F
            frac = data[2] / 256.0
            self.battery_voltage = round(coarse + frac, 2)
            state_map = {1: "New", 2: "Good", 3: "Ok", 4: "Low", 5: "Critical"}
            self.battery_status = state_map.get((data[3] & 0x70) >> 4, "Unknown")

        elif page == 2:  # Manufacturer
            self.manufacturer_id = data[1]
            self.serial_number = (data[3] << 8) | data[2]

        elif page == 1:  # Cumulative Operating Time
            # 3 bytes, 2-second resolution
            cumulative_secs = (data[1] | (data[2] << 8) | (data[3] << 16)) * 2
            self.operating_time_hours = round(cumulative_secs / 3600.0, 1)

        # 2. Parse Heart Rate
        self.bpm = data[7]

        # 3. Parse RR Intervals
        beat_count = data[6]
        beat_time_raw = (data[5] << 8) | data[4]
        beat_time = beat_time_raw / 1024.0

        if self.last_beat_time is not None:
            if beat_time < self.last_beat_time: beat_time += 64.0

            if beat_count != self.last_beat_count:
                delta = beat_time - self.last_beat_time
                self.raw_rr_ms = int(delta * 1000)

                # Filter dropped packets
                if delta > 1.5:
                    self.filter_buffer.clear()
                    self.last_beat_time = beat_time_raw / 1024.0
                    self.last_beat_count = beat_count
                    return

                if self._is_valid_beat(delta):
                    self.rr_ms = self.raw_rr_ms
                    self.rr_buffer.append(self.rr_ms)
                    self.filter_buffer.append(delta)
                    self.rmssd = self._calculate_rmssd_safe()
                    self.status = "Active"

        self.last_beat_time = beat_time_raw / 1024.0
        self.last_beat_count = beat_count

    def _is_valid_beat(self, rr_sec):
        if rr_sec < 0.27 or rr_sec > 1.5: return False
        if len(self.filter_buffer) > 0:
            avg = sum(self.filter_buffer) / len(self.filter_buffer)
            if abs(rr_sec - avg) > (avg * 0.3):
                self.consecutive_rejections += 1
                return False
        self.consecutive_rejections = 0
        return True

    def _calculate_rmssd_safe(self):
        if len(self.rr_buffer) < 2: return 0.0
        try:
            diffs = [self.rr_buffer[i] - self.rr_buffer[i - 1] for i in range(1, len(self.rr_buffer))]
            sq_diffs = [d * d for d in diffs]
            return math.sqrt(sum(sq_diffs) / len(sq_diffs))
        except:
            return 0.0

    def _run_loop(self):
        try:
            self.node = Node()
            self.node.set_network_key(0, ANTPLUS_NETWORK_KEY)

            # --- CHANNEL 0: HEART RATE (Wildcard) ---
            self.channel_hr = self.node.new_channel(Channel.Type.BIDIRECTIONAL_RECEIVE)
            self.channel_hr.on_broadcast_data = self._on_hr_data
            self.channel_hr.on_burst_data = self._on_hr_data
            self.channel_hr.set_id(0, 0, 0)  # Wildcard HR
            self.channel_hr.set_rf_freq(57)
            self.channel_hr.set_period(8070)
            self.channel_hr.open()

            self.node.start()

        except Exception as e:
            self.status = f"Error: {e}"
        finally:
            self.running = False
            if self.channel_hr:
                try: self.channel_hr.close();
                except: pass
            if self.node:
                try: self.node.stop();
                except: pass