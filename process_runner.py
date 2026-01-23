import json
import os
import sys
import hashlib
from gtts import gTTS
from pathlib import Path

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class ProcessRunner:
    def __init__(self, processes_file):
        """Initialize the ProcessRunner with a JSON configuration file."""
        self.processes_file = processes_file
        self.processes = {}
        self.mixer_initialized = False
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init()
            self.mixer_initialized = True
        except Exception as e:
            print(f"Warning: Audio initialization failed. Audio will not play. Error: {e}", flush=True)
            
        self.load_processes()

    def load_processes(self):
        """Load processes from the JSON file."""
        try:
            with open(self.processes_file, 'r') as f:
                data = json.load(f)
                # Convert list to dict for easier lookup by name
                for p in data.get('processes', []):
                    self.processes[p['name']] = p
        except FileNotFoundError:
            print(f"Error: Configuration file '{self.processes_file}' not found.", flush=True)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{self.processes_file}'.", flush=True)

    def list_processes(self):
        """Return a list of available process names."""
        return list(self.processes.keys())

    def get_process_data(self, process_name):
        """Return raw data for a process."""
        return self.processes.get(process_name)

    def prepare_step_audio(self, text, audio_file):
        """Pre-generates or locates audio for a step. Returns path or None."""
        # 1. Use existing audio file if valid
        if audio_file and os.path.exists(audio_file):
            return audio_file
        
        # 2. Fallback to TTS Generation
        if text:
            tts = TTSEngine()
            return tts.generate_audio(text)
            
        return None

    def play_audio_file(self, file_path):
        """Non-blocking playback request to pygame."""
        if self.mixer_initialized and file_path:
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                return True
            except Exception as e:
                print(f"Error playing audio '{file_path}': {e}", flush=True)
                return False
        return False
        
    def stop_audio(self):
        if self.mixer_initialized:
            pygame.mixer.music.stop()

    def is_playing(self):
        if self.mixer_initialized:
            return pygame.mixer.music.get_busy()
        return False

    def run_process_cli(self, process_name):
        """Run the specified process (Blocking CLI Mode)."""
        if process_name not in self.processes:
            print(f"Process '{process_name}' not found.", flush=True)
            return

        process = self.processes[process_name]
        print(f"\n--- Starting Process: {process['name']} ---", flush=True)
        
        for i, step in enumerate(process['steps']):
            text = step.get('text', '')
            audio_file = step.get('audio_file', '')
            
            # Display text
            print(f"\n[Step {i+1}] {text}", flush=True)
            
            # Play
            file_to_play = self.prepare_step_audio(text, audio_file)
            
            if file_to_play:
                print(f"DEBUG: Playing {file_to_play}...", flush=True)
                self.play_audio_file(file_to_play)
                
                while self.is_playing():
                     pygame.time.Clock().tick(10)
                print("DEBUG: Playback finished.", flush=True)

            # Wait for user confirmation
            input(">> Press Enter after answering...")
            
        print(f"\n--- Process {process['name']} Complete ---", flush=True)


class TTSEngine:
    def __init__(self):
        self.cache_dir = "tts_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def generate_audio(self, text):
        """Generates audio for text using gTTS, using cache if available."""
        if not text: return None
        
        # Create hash filename
        filename = hashlib.md5(text.encode()).hexdigest() + ".mp3" # gTTS saves as mp3
        file_path = os.path.join(self.cache_dir, filename)
        
        # Check cache
        if os.path.exists(file_path):
            print(f"DEBUG: Using cached TTS: {file_path}", flush=True)
            return file_path
            
        try:
            print("DEBUG: Requesting gTTS...", flush=True)
            tts = gTTS(text=text, lang='en', tld='co.uk')
            tts.save(file_path)
            print("DEBUG: gTTS saved.", flush=True)
            return file_path
        except Exception as e:
            print(f"gTTS Request Failed: {e}", flush=True)
            return None

if __name__ == "__main__":
    print("Starting Process Runner v2.1 (Debug)...", flush=True)
    # Simple CLI for testing
    runner = ProcessRunner("processes.json")
    available = runner.list_processes()
    
    if not available:
        print("No processes available.", flush=True)
    else:
        print("Available Processes:", flush=True)
        for idx, name in enumerate(available):
            print(f"{idx}: {name}", flush=True)
            
        choice = input("\nSelect process index (or name): ").strip()
        
        selected_name = None
        if choice.isdigit() and int(choice) < len(available):
            selected_name = available[int(choice)]
        elif choice in available:
            selected_name = choice
            
        if selected_name:
            runner.run_process(selected_name)
        else:
            print("Invalid selection.", flush=True)
