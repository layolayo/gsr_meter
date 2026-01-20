import sounddevice as sd
import soundfile as sf
import time
import os
import numpy as np

# Path from user logs
FILE_PATH = "/home/matthew/PycharmProjects/BrainCo/Session_Data/GSR_Audio_2026-01-02_21-54-04.wav"

def test_playback():
    print(f"Testing Audio Loading from: {FILE_PATH}")
    
    if not os.path.exists(FILE_PATH):
        print("ERROR: File not found!")
        return

    # Load
    try:
        data, fs = sf.read(FILE_PATH, dtype='float32')
        print(f"Loaded: {len(data)} samples, {fs}Hz")
        print(f"Shape: {data.shape}")
        print(f"Stats: Min={data.min()}, Max={data.max()}")
    except Exception as e:
        print(f"Load Error: {e}")
        return

    # Play
    print("\nAttempting Playback (Async - Main Thread)...")
    print("You should hear audio immediately.")
    
    try:
        sd.play(data, fs)
        
        # Wait for duration
        duration = len(data) / fs
        print(f"Playing for {duration:.1f} seconds (Press Ctrl+C to stop)...")
        time.sleep(duration)
        print("Playback finished.")
        
    except Exception as e:
        print(f"Play Error: {e}")

if __name__ == "__main__":
    test_playback()
