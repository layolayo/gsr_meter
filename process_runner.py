import asyncio
import edge_tts
import json
import os
import sys
import hashlib
import copy
from gtts import gTTS # Keeping for fallback if needed, though edge-tts is primary now
from pathlib import Path

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# [NEW] Safety Constants
INFINITE_REPEAT_DEFAULT = 100
INNER_INFINITE_REPEAT_DEFAULT = 25
SAFE_MAX_STEPS = 10000

class ProcessRunner:
    def __init__(self, processes_file):
        """Initialize the ProcessRunner with a JSON configuration file."""
        self.processes_file = processes_file
        self.assessments_file = "assessments.json"
        self.processes = {}
        self.raw_processes = {} # [NEW] Store raw for dynamic re-compilation
        self.assessments = {}
        self.starting_questions = [] # [NEW]
        self.closing_questions = [] 
        self.mixer_initialized = False
        
        # Audio Settings
        self.voice_tld = "co.uk"    # Default accent
        self.voice_gender = "Female" # [NEW] Default gender
        self.audio_enabled = True   # Global toggle for current session
        
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
                self.question_library = data.get('question_library', {})
                
                # [NEW] Load Assessments
                try: 
                    with open(self.assessments_file, 'r') as fa:
                        self.assessments = json.load(fa)
                except: self.assessments = {}

                # Convert list to dict for easier lookup by name
                for p in data.get('processes', []):
                    # Store raw copy first
                    self.raw_processes[p['name']] = copy.deepcopy(p)
                    
                    
                    # [NEW] Check for mixed format in Steps (Convert keys to objects)
                    if 'steps' in p:
                        norm_steps = []
                        for s in p['steps']:
                            if isinstance(s, str):
                                # Key lookup
                                if s in self.question_library:
                                    norm_steps.append(self.question_library[s].copy())
                                else:
                                    print(f"Warning: Step Key '{s}' not found.", flush=True)
                            elif isinstance(s, dict):
                                # Check for Inline Structure (Recursive Compile)
                                if 'pattern' in s or 'iterate_assessment' in s:
                                     sub_steps = self.compile_process(s)
                                     norm_steps.extend(sub_steps)
                                else:
                                     norm_steps.append(s)
                        p['steps'] = norm_steps
                        
                    self.processes[p['name']] = p
                
                # [NEW] Load Starting Questions
                raw_starting = data.get('starting_questions', [])
                self.starting_questions = self._load_aux_questions(raw_starting)
                
                # [NEW] Load Closing Questions
                raw_closing = data.get('closing_questions', [])
                self.closing_questions = self._load_aux_questions(raw_closing)
                



                
                print(f"[ProcessRunner] Loaded {len(self.starting_questions)} starting questions.", flush=True)
                print(f"[ProcessRunner] Loaded {len(self.closing_questions)} closing questions.", flush=True)
        except FileNotFoundError:
            print(f"Error: Configuration file '{self.processes_file}' not found.", flush=True)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{self.processes_file}'.", flush=True)

    def _load_aux_questions(self, raw_data):
        """Helper to load starting/closing questions handles list or dict format."""
        steps = []
        # Normalize input to list of items
        items = []
        if isinstance(raw_data, dict) and 'steps' in raw_data:
             items = raw_data['steps']
        elif isinstance(raw_data, list):
             # Check if it's a list of objects or a single object wrapper
             for i in raw_data:
                  if isinstance(i, dict) and 'steps' in i:
                       items.extend(i['steps'])
                  else:
                       items.append(i)
        
        for s in items:
             if isinstance(s, str):
                   if s in self.question_library:
                        steps.append(self.question_library[s].copy())
                   else:
                        print(f"Warning: Aux Step Key '{s}' not found.")
             elif isinstance(s, dict):
                   if 'pattern' in s or 'iterate_assessment' in s:
                        # Recursive Compile
                        sub = self.compile_process(s)
                        steps.extend(sub)
                   else:
                        steps.append(s)
        return steps

    def get_required_assessments(self, process_name):
        """Returns a list of assessment keys used by the process."""
        if process_name not in self.raw_processes: return []
        
        raw = self.raw_processes[process_name]
        required = set()
        
        def scan_struct(s):
            if isinstance(s, dict):
                if 'iterate_assessment' in s:
                    required.add(s['iterate_assessment'])
                # Recurse
                for k, v in s.items():
                    scan_struct(v)
            elif isinstance(s, list):
                for i in s:
                    scan_struct(i)
                    
        scan_struct(raw)
        return list(required)

    def compile_process_dynamic(self, process_name, selection_map):
        """Re-compiles a process using user-selected assessment items."""
        if process_name not in self.raw_processes: return []
        
        # Get raw definition
        p = copy.deepcopy(self.raw_processes[process_name])
        
        # Helper to compile with selection context
        def compile_recursive(struct):
            return self.compile_process(struct, selection_map)

        # Standard compilation logic but using new selection map
            
        if 'steps' in p:
            norm_steps = []
            for s in p['steps']:
                if isinstance(s, str):
                     if s in self.question_library:
                          norm_steps.append(self.question_library[s].copy())
                elif isinstance(s, dict):
                     if 'pattern' in s or 'iterate_assessment' in s:
                          norm_steps.extend(self.compile_process(s, selection_map))
                     else:
                          norm_steps.append(s)
            p['steps'] = norm_steps
            
        return p['steps']

    def compile_process(self, structure, selection_map=None, substitution_map=None, depth=0, parent_loop_id=None, usage_counts=None):
        """Compiles a process list from a structure definition. Supports recursion."""
        steps = []
        if substitution_map is None:
            substitution_map = {}
        if usage_counts is None:
             usage_counts = {} # Key: Question Key, Value: Count
        
        # Determine effective "infinity"
        inf_limit = INFINITE_REPEAT_DEFAULT if depth == 0 else INNER_INFINITE_REPEAT_DEFAULT
        
        # Check for Assessment Iteration
        iter_key = structure.get('iterate_assessment')
        if iter_key and iter_key in self.assessments:
             # [MOD] CHECK SELECTION MAP
             assess_list = self.assessments[iter_key]
             if selection_map and iter_key in selection_map:
                  sel = selection_map[iter_key]
                  if isinstance(sel, str): assess_list = [sel]
                  elif isinstance(sel, list): assess_list = sel
             
             if structure.get('shuffle', False):
                  import random
                  random.shuffle(assess_list)
             
             pattern = structure.get('pattern', [])
             repeat_count = structure.get('repeat', 1) 
             is_break_requested = structure.get('is_break_prompt', False)
             
             if repeat_count == -1: repeat_count = inf_limit
             
             import uuid
             my_uuid = str(uuid.uuid4())[:8] if is_break_requested else None
             effective_loop_id = (parent_loop_id + "." if parent_loop_id else "") + my_uuid if my_uuid else parent_loop_id
             
             for r in range(repeat_count):
                 total_pattern_steps = len(assess_list) * len(pattern)
                 current_pattern_idx = 0
                 for idx, item in enumerate(assess_list):
                      # Update substitution map for this item
                      local_substitution = substitution_map.copy()
                      local_substitution[iter_key] = item
                      
                      for pat_item in pattern:
                           current_pattern_idx += 1
                           
                           if isinstance(pat_item, dict) and ('pattern' in pat_item or 'iterate_assessment' in pat_item):
                                # Recursive call for nested patterns
                                sub_steps = self.compile_process(pat_item, selection_map, local_substitution, depth=depth+1, parent_loop_id=effective_loop_id, usage_counts=usage_counts)
                                steps.extend(sub_steps)
                                if len(steps) >= SAFE_MAX_STEPS:
                                     print(f"Warning: Process exceeded {SAFE_MAX_STEPS} steps. Capping compilation.", flush=True)
                                     return steps[:SAFE_MAX_STEPS]
                                continue

                           # Resolve Key/Step
                           base_step = None
                           q_key = None
                           if isinstance(pat_item, str) and pat_item in self.question_library:
                                q_key = pat_item
                                base_step = self.question_library[pat_item].copy()
                           elif isinstance(pat_item, dict):
                                q_key = pat_item.get('key')
                                if q_key and q_key in self.question_library:
                                     base_step = self.question_library[q_key].copy()
                                     base_step.update({k:v for k,v in pat_item.items() if k != 'key'})
                                else:
                                     base_step = pat_item.copy()
                           
                           if base_step:
                                # Perform Substitution
                                txt = base_step.get('text', "")
                                for skey, sval in local_substitution.items():
                                     txt = txt.replace(f"[{skey}]", sval)
                                # [NEW] Dynamic [else] Substitution
                                if q_key:
                                     usage_counts[q_key] = usage_counts.get(q_key, 0) + 1
                                     else_str = "else " if usage_counts[q_key] > 1 else ""
                                     txt = txt.replace("[else]", else_str)
                                     
                                base_step['text'] = txt
                                
                                # Metadata
                                base_step['set'] = str(idx + 1)
                                base_step['assessment_item'] = item
                                
                                if is_break_requested:
                                     base_step['loop_id'] = effective_loop_id
                                     if current_pattern_idx == total_pattern_steps:
                                          base_step['is_break_prompt'] = True
                                          
                                steps.append(base_step)
                                if len(steps) >= SAFE_MAX_STEPS:
                                     print(f"Warning: Process exceeded {SAFE_MAX_STEPS} steps. Capping compilation.", flush=True)
                                     return steps
             return steps

        # Standard Repeat Logic (no iterate_assessment at this level)
        pattern = structure.get('pattern', [])
        repeat = structure.get('repeat', 1)
        is_break_requested = structure.get('is_break_prompt', False)
        
        if repeat == -1: repeat = inf_limit

        import uuid
        my_uuid = str(uuid.uuid4())[:8] if is_break_requested else None
        effective_loop_id = (parent_loop_id + "." if parent_loop_id else "") + my_uuid if my_uuid else parent_loop_id
        
        for r in range(repeat):
            set_num = r + 1
            for idx, pat_item in enumerate(pattern):
                if isinstance(pat_item, dict) and ('pattern' in pat_item or 'iterate_assessment' in pat_item):
                     # Recursive call for nested patterns
                     sub_steps = self.compile_process(pat_item, selection_map, substitution_map, depth=depth+1, parent_loop_id=effective_loop_id, usage_counts=usage_counts)
                     steps.extend(sub_steps)
                     if len(steps) >= SAFE_MAX_STEPS:
                          print(f"Warning: Process exceeded {SAFE_MAX_STEPS} steps. Capping compilation.", flush=True)
                          return steps[:SAFE_MAX_STEPS]
                     continue

                step = None
                q_key = None
                if isinstance(pat_item, str) and pat_item in self.question_library:
                    q_key = pat_item
                    step = self.question_library[pat_item].copy()
                elif isinstance(pat_item, dict):
                    q_key = pat_item.get('key')
                    if q_key and q_key in self.question_library:
                         step = self.question_library[q_key].copy()
                         step.update({k:v for k,v in pat_item.items() if k != 'key'})
                    else:
                         step = pat_item.copy()
                    
                if step:
                    # Substitution
                    txt = step.get('text', "")
                    for skey, sval in substitution_map.items():
                         txt = txt.replace(f"[{skey}]", sval)
                    
                    # [NEW] Dynamic [else] Substitution
                    if q_key:
                         usage_counts[q_key] = usage_counts.get(q_key, 0) + 1
                         else_str = "else " if usage_counts[q_key] > 1 else ""
                         txt = txt.replace("[else]", else_str)
                         
                    step['text'] = txt
                    
                    if effective_loop_id:
                         step['loop_id'] = effective_loop_id
                         if is_break_requested and idx == len(pattern) - 1:
                              step['is_break_prompt'] = True
                              
                    steps.append(step)
                    if len(steps) >= SAFE_MAX_STEPS:
                         print(f"Warning: Process exceeded {SAFE_MAX_STEPS} steps. Capping compilation.", flush=True)
                         return steps
                else:
                    if isinstance(pat_item, str):
                         print(f"Warning: Question key '{pat_item}' not found in library.", flush=True)
        return steps

    def list_processes(self):
        """Return a list of available process names."""
        return list(self.processes.keys())

    def get_process_data(self, process_name):
        """Return raw data for a process."""
        return self.processes.get(process_name)

    def get_closing_questions(self):
        """Return the list of closing questions."""
        return self.closing_questions

    def prepare_step_audio(self, text, audio_file):
        """Pre-generates or locates audio for a step. Returns path or None."""
        if not self.audio_enabled:
            return None
            
        # 1. Use existing audio file if valid
        if audio_file and os.path.exists(audio_file):
            return audio_file
        
        # 2. Fallback to TTS Generation
        if text:
            tts = TTSEngine()
            return tts.generate_audio(text, tld=self.voice_tld, gender=self.voice_gender)
            
        return None

    def _check_mixer(self):
        """Ensure mixer is initialized, attempt re-init if needed."""
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
                self.mixer_initialized = True
            except:
                self.mixer_initialized = False
        else:
            self.mixer_initialized = True
        return self.mixer_initialized

    def play_audio_file(self, file_path):
        """Non-blocking playback request to pygame."""
        if self._check_mixer() and file_path:
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                return True
            except Exception as e:
                print(f"Error playing audio '{file_path}': {e}", flush=True)
                return False
        return False
        
    def stop_audio(self):
        if self._check_mixer():
            pygame.mixer.music.stop()

    def is_playing(self):
        if self._check_mixer():
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

    def generate_audio(self, text, tld="co.uk", gender="Female"):
        """Generates audio for text using edge-tts (neural voices)."""
        if not text: return None
        
        # Mapping for edge-tts voices
        # Locale -> {Male: VoiceID, Female: VoiceID}
        voice_map = {
            "co.uk":  {"Male": "en-GB-ThomasNeural", "Female": "en-GB-SoniaNeural"},
            "com":     {"Male": "en-US-AndrewMultilingualNeural", "Female": "en-US-JennyNeural"},
            "com.au":  {"Male": "en-AU-WilliamMultilingualNeural", "Female": "en-AU-NatashaNeural"},
            "co.in":   {"Male": "en-IN-PrabhatNeural", "Female": "en-IN-NeerjaNeural"},
            "ca":      {"Male": "en-CA-LiamNeural", "Female": "en-CA-ClaraNeural"},
            "ie":      {"Male": "en-IE-ConnorNeural", "Female": "en-IE-EmilyNeural"},
            "nz":      {"Male": "en-NZ-MitchellNeural", "Female": "en-NZ-MollyNeural"},
            "za":      {"Male": "en-ZA-LukeNeural", "Female": "en-ZA-LeahNeural"}
        }
        
        voice_id = voice_map.get(tld, voice_map["co.uk"]).get(gender, "en-GB-SoniaNeural")
        
        # Create hash filename (include voice_id in hash to separate variants in cache)
        cache_key = f"{text}|{voice_id}"
        filename = hashlib.md5(cache_key.encode()).hexdigest() + ".mp3"
        file_path = os.path.join(self.cache_dir, filename)
        
        # Check cache
        if os.path.exists(file_path):
            print(f"DEBUG: Using cached Edge-TTS [{voice_id}]: {file_path}", flush=True)
            return file_path
            
        try:
            print(f"DEBUG: Requesting Edge-TTS (Voice: {voice_id})...", flush=True)
            
            async def _generate():
                communicate = edge_tts.Communicate(text, voice_id)
                await communicate.save(file_path)
            
            asyncio.run(_generate())
            
            if os.path.exists(file_path):
                print("DEBUG: Edge-TTS saved.", flush=True)
                return file_path
        except Exception as e:
            print(f"Edge-TTS Request Failed: {e}. Falling back to gTTS...", flush=True)
            # Fallback to gTTS if Edge-TTS fails
            try:
                tts = gTTS(text=text, lang='en', tld=tld if '.' in tld else 'co.uk')
                tts.save(file_path)
                return file_path
            except: pass
            
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
