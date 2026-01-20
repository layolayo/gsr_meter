import asyncio
import logging
import time
import struct
from datetime import datetime
from bleak import BleakClient, BleakScanner

# --- CONFIGURATION ---
DEVICE_ADDRESS = "58:94:b2:00:b4:35"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILENAME = f"brainco_report_{TIMESTAMP}.txt"

# --- UUIDS ---
UUID_BATTERY = "00002a19-0000-1000-8000-00805f9b34fb"
UUID_SERIAL = "00002a25-0000-1000-8000-00805f9b34fb"
UUID_MANUFACTURER = "00002a29-0000-1000-8000-00805f9b34fb"
UUID_MODEL_NUMBER = "00002a24-0000-1000-8000-00805f9b34fb"  # <--- Added Missing UUID
UUID_FIRMWARE = "00002a26-0000-1000-8000-00805f9b34fb"
UUID_HARDWARE = "00002a27-0000-1000-8000-00805f9b34fb"

# BrainCo Specifics
UUID_WRITE = "0d740002-d26f-4dbb-95e8-a4f5c55c57a9"
UUID_NOTIFY = "0d740003-d26f-4dbb-95e8-a4f5c55c57a9"

# Commands
AUTH_COMMAND = bytearray(
    [0x43, 0x4d, 0x53, 0x4e, 0x00, 0x18, 0x08, 0x01, 0x12, 0x14, 0x08, 0x01, 0x32, 0x10, 0x30, 0x35, 0x32, 0x31, 0x65,
     0x32, 0x62, 0x64, 0x34, 0x38, 0x62, 0x62, 0x63, 0x62, 0x66, 0x36, 0x50, 0x4b, 0x45, 0x44])
START_COMMAND = bytearray(
    [0x43, 0x4d, 0x53, 0x4e, 0x00, 0x08, 0x08, 0x01, 0x32, 0x04, 0x10, 0x01, 0x18, 0x00, 0x50, 0x4b, 0x45, 0x44])
STREAM_COMMAND = bytearray(
    [0x43, 0x4d, 0x53, 0x4e, 0x00, 0x07, 0x08, 0x03, 0x12, 0x03, 0x0a, 0x01, 0x03, 0x50, 0x4b, 0x45, 0x44])

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Diag")

# --- STATE ---
stream_stats = {
    "start_time": 0,
    "first_packet_time": 0,
    "packet_count": 0,
    "total_bytes": 0,
    "payload_lengths": {}
}


def notification_handler(sender, data):
    """Analyzes incoming data packets without decoding full EEG."""
    now = time.time()

    # 1. Latency Check
    if stream_stats["first_packet_time"] == 0:
        stream_stats["first_packet_time"] = now
        latency = (now - stream_stats["start_time"]) * 1000
        logger.info(f"[STREAM] First Data Received! Latency: {latency:.0f}ms")

    stream_stats["packet_count"] += 1

    # 2. Structure Analysis
    if data.startswith(b'CMSN'):
        body = data[8:]
        idx = 0
        while idx < len(body):
            if idx >= len(body): break
            key = body[idx]
            idx += 1

            # Get Length
            if idx >= len(body): break
            length = body[idx]
            idx += 1
            if length & 0x80:
                length = (length & 0x7F) | (body[idx] << 7)
                idx += 1

            # Record Payload Size for Analysis
            hex_key = f"0x{key:02X}"
            if hex_key not in stream_stats["payload_lengths"]:
                stream_stats["payload_lengths"][hex_key] = []

            # Store first 5 unique lengths found per key
            if length not in stream_stats["payload_lengths"][hex_key]:
                stream_stats["payload_lengths"][hex_key].append(length)

            idx += length


async def run_diagnostics():
    logger.info(f"=== BRAINCO DIAGNOSTIC TOOL v1.1 ===")
    logger.info(f"Target Address: {DEVICE_ADDRESS}")

    # 1. CONNECTION
    logger.info("\n[PHASE 1] CONNECTING...")
    t0 = time.time()

    try:
        async with BleakClient(DEVICE_ADDRESS, timeout=20.0) as client:
            t1 = time.time()
            logger.info(f"  -> Connected in {t1 - t0:.2f} seconds.")

            # 2. IDENTITY
            logger.info("\n[PHASE 2] DEVICE IDENTITY")
            for name, uuid in [
                ("Manufacturer", UUID_MANUFACTURER),
                ("Model Number", UUID_MODEL_NUMBER),
                ("Serial Number", UUID_SERIAL),
                ("Firmware Ver", UUID_FIRMWARE),
                ("Hardware Ver", UUID_HARDWARE)
            ]:
                try:
                    val = await client.read_gatt_char(uuid)
                    logger.info(f"  -> {name}: {val.decode('utf-8')}")
                except:
                    logger.info(f"  -> {name}: [Read Failed / Not Supported]")

            # 3. AUTHENTICATION & BATTERY
            logger.info("\n[PHASE 3] AUTHENTICATION")
            logger.info("  -> Sending Magic Bytes...")
            await client.write_gatt_char(UUID_WRITE, AUTH_COMMAND[0:20], response=False)
            await asyncio.sleep(0.05)
            await client.write_gatt_char(UUID_WRITE, AUTH_COMMAND[20:], response=False)

            logger.info("  -> Waiting 1.5s for Unlock...")
            await asyncio.sleep(1.5)

            logger.info("\n[PHASE 4] BATTERY STATUS")
            try:
                val = await client.read_gatt_char(UUID_BATTERY)
                # Battery usually comes as a single byte (percentage)
                pct = int.from_bytes(val, byteorder='little')
                logger.info(f"  -> Battery Level: {pct}%")
            except Exception as e:
                logger.error(f"  -> Battery Read Failed: {e}")

            # 4. STREAM ANALYSIS
            logger.info("\n[PHASE 5] STREAM ANALYSIS")
            await client.start_notify(UUID_NOTIFY, notification_handler)

            logger.info("  -> Sending Start Command...")
            await client.write_gatt_char(UUID_WRITE, START_COMMAND, response=False)
            await asyncio.sleep(0.5)

            logger.info("  -> Sending Stream Command...")
            stream_stats["start_time"] = time.time()
            await client.write_gatt_char(UUID_WRITE, STREAM_COMMAND, response=False)

            logger.info("  -> Collecting data for 5 seconds...")
            await asyncio.sleep(5.0)

            await client.stop_notify(UUID_NOTIFY)

            # 5. RESULTS
            logger.info("\n=== ANALYSIS RESULTS ===")
            duration = time.time() - stream_stats["start_time"]
            pps = stream_stats["packet_count"] / duration

            logger.info(f"Packets Received: {stream_stats['packet_count']}")
            logger.info(f"Duration: {duration:.2f}s")
            logger.info(f"Packet Rate: {pps:.2f} packets/sec")

            logger.info("\nPacket Structure Found:")
            for key, lengths in stream_stats["payload_lengths"].items():
                desc = "UNKNOWN"
                if key == "0x22":
                    desc = "EEG DATA"
                elif key == "0x08":
                    desc = "SIGNAL QUALITY"
                elif key == "0x10":
                    desc = "HEADSET ON/OFF"

                logger.info(f"  Key {key} ({desc}): Payload Lengths found = {lengths}")

                if key == "0x22":
                    for l in lengths:
                        if l % 3 == 0:
                            logger.info(f"    -> Length {l} is divisible by 3. (Supports 24-bit: {l // 3} samples)")
                        if l % 2 == 0:
                            logger.info(f"    -> Length {l} is divisible by 2. (Supports 16-bit: {l // 2} samples)")

    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_diagnostics())