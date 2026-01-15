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
        self.bpm = 0
        self.rr_ms = 0
        self.rmssd = 0.0
        self.status = "Initializing"

        # New Raw Data Fields
        self.raw_rr_ms = 0
        self.last_packet_hex = ""

        self.rr_buffer = collections.deque(maxlen=30)
        self.filter_buffer = collections.deque(maxlen=5)
        self.consecutive_rejections = 0

        self.node = None
        self.channel = None
        self.last_beat_time = None
        self.last_beat_count = None
        self.last_data_time = time.time()

    def start(self):
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

    def get_data(self):
        time_since_update = time.time() - self.last_data_time
        status_out = self.status
        if time_since_update > 2.0 and self.status == "Active":
            status_out = "No Signal"

        return {
            "bpm": self.bpm,
            "rr_ms": self.rr_ms,
            "raw_rr_ms": self.raw_rr_ms,
            "rmssd": self.rmssd,
            "status": status_out,
            "staleness": time_since_update,
            "raw_hex": self.last_packet_hex
        }

    def _calculate_rmssd_safe(self):
        if len(self.rr_buffer) < 2:
            return 0.0
        rr_list = list(self.rr_buffer)
        diffs = [rr_list[i + 1] - rr_list[i] for i in range(len(rr_list) - 1)]
        squared_diffs = [d ** 2 for d in diffs]
        return math.sqrt(sum(squared_diffs) / len(squared_diffs))

    def _is_valid_beat(self, new_rr_sec):
        if len(self.filter_buffer) < 3:
            return 0.27 < new_rr_sec < 1.5  # Tighter upper bound

        if self.consecutive_rejections >= 3:
            self.filter_buffer.clear()
            self.consecutive_rejections = 0
            return True

        local_avg = sum(self.filter_buffer) / len(self.filter_buffer)
        threshold = 0.30
        min_valid = local_avg * (1.0 - threshold)
        max_valid = local_avg * (1.0 + threshold)

        if min_valid < new_rr_sec < max_valid:
            self.consecutive_rejections = 0
            return True
        else:
            self.consecutive_rejections += 1
            return False

    def _on_data(self, data):
        self.last_data_time = time.time()
        self.last_packet_hex = "".join('{:02x}'.format(x) for x in data)

        try:
            self.channel.request_message(0x51)
        except:
            pass

        page = data[0] & 0x7F
        if page == 4 or page == 0:
            beat_time = struct.unpack('<H', bytes(data[4:6]))[0] / 1024.0
            beat_count = data[6]
            self.bpm = data[7]

            if self.last_beat_time is not None and beat_count != self.last_beat_count:
                delta = beat_time - self.last_beat_time
                if delta < 0: delta += 64.0

                self.raw_rr_ms = int(delta * 1000)

                # --- FIX: THE DISCONNECT FILTER ---
                # If a beat takes > 1.5 seconds (40 bpm), it's likely a dropped packet.
                # Do NOT calculate RMSSD for this gap.
                if delta > 1.5:
                    self.filter_buffer.clear()  # Reset rolling average
                    # Update time but SKIP adding to rr_buffer
                    self.last_beat_time = beat_time
                    self.last_beat_count = beat_count
                    return

                if self._is_valid_beat(delta):
                    self.rr_ms = self.raw_rr_ms
                    self.rr_buffer.append(self.rr_ms)
                    self.filter_buffer.append(delta)
                    self.rmssd = self._calculate_rmssd_safe()
                    self.status = "Active"

            self.last_beat_time = beat_time
            self.last_beat_count = beat_count

    def _run_loop(self):
        try:
            self.node = Node()
            self.node.set_network_key(0, ANTPLUS_NETWORK_KEY)
            self.channel = self.node.new_channel(Channel.Type.BIDIRECTIONAL_RECEIVE)
            self.channel.on_broadcast_data = self._on_data
            self.channel.on_burst_data = self._on_data
            self.channel.set_id(0, 0, 0)
            self.channel.set_rf_freq(57)
            self.channel.set_period(8070)
            self.channel.open()
            self.node.start()
        except Exception as e:
            self.status = "Error"