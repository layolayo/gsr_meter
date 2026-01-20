
import csv
import math
import os
import sys
import glob

# Constants from v22 logic
MAX_SWEEP = 40
RESET_TARGET = 11

def calculate_angle(ta, center, sens):
    diff = ta - center
    zoom = sens if sens > 0 else 0.2
    half_win = zoom / 2.0
    
    ratio = diff / half_win
    # Clamp ratio for visual check logic (same as viewer)
    # logic: angle_v = (ratio * 40) + 11
    
    # We want to see the RAW calculated angle
    angle_v = (ratio * MAX_SWEEP) + RESET_TARGET
    return angle_v

def analyze_session(folder_path):
    gsr_path = os.path.join(folder_path, "gsr.csv")
    if not os.path.exists(gsr_path):
        print(f"No gsr.csv found in {folder_path}")
        return

    print(f"Analyzing: {gsr_path}")
    
    with open(gsr_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if not rows:
        print("Empty file")
        return

    print(f"Total Rows: {len(rows)}")
    
    last_center = None
    last_sens = None
    resets_found = 0
    ok_resets = 0
    
    for i, row in enumerate(rows):
        try:
            t = float(row['Elapsed_Sec'])
            ta = float(row['GSR_TA'])
            center = float(row['GSR_SetPoint'])
            sens = float(row['GSR_Sens'])
        except ValueError:
            continue
            
        angle = calculate_angle(ta, center, sens)
        
        # Check for change in Center/Sens (Reset Event)
        if last_center is not None and (center != last_center or sens != last_sens):
            # Reset Detected at this frame
            resets_found += 1
            
            # Previous Frame
            prev_row = rows[i-1]
            prev_ta = float(prev_row['GSR_TA'])
            prev_center = float(prev_row['GSR_SetPoint'])
            prev_sens = float(prev_row['GSR_Sens'])
            prev_angle = calculate_angle(prev_ta, prev_center, prev_sens)
            
            # Check correctness
            # Pre-Reset: Should be > 40 or < -40 (or close to edge) relative to vertical?
            # Wait, our Angle logic (ratio*40)+11 implies limits are "Visual Limits".
            # Viewer Logic: Visual limits are +/- 40 from vertical.
            # Angle V calculation returns angle relative to vertical (0 = 12 o'clock)?
            # No, Angle V is offset from Center? 
            # In Viewer: rad = math.radians(ANGLE_CENTER + angle_v)
            # ANGLE_CENTER = 90.
            # So angle_v IS degrees from 12 o'clock (Right is negative? Left is positive?)
            # Wait. math.cos(90+angle). 
            # If angle = 0 -> 90 -> Up.
            # If angle = 40 -> 130 -> Left.
            # If angle = -40 -> 50 -> Right.
            
            # User expectation:
            # Pre-Reset: Angle is at/past limit (+40 or -40).
            # Post-Reset: Angle is at SET Position (11).
            
            is_good = False
            if abs(angle - 11) < 5.0: # Tolerance of 5 degrees
                is_good = True
                ok_resets += 1
                
            status = "GOOD" if is_good else "FAIL"
            
            print(f"RESET #{resets_found} at T={t:.3f}s")
            print(f"  PRE : Angle={prev_angle:.1f} (Limit +/-40?) | TA={prev_ta:.4f} | Center={prev_center:.4f}")
            motion = int(row.get('GSR_Motion', 0))
            print(f"  POST: Angle={angle:.1f} (Target 11.0)  | TA={ta:.4f}     | Center={center:.4f} | Motion={motion}")
            print(f"  Status: {status}")
            print("-" * 40)
            
        last_center = center
        last_sens = sens
        
    print(f"Analysis Complete. Found {resets_found} resets. {ok_resets} aligned to 11 +/- 5 deg.")

def main():
    # Find latest session
    base_dir = "Session_Data"
    if not os.path.exists(base_dir):
        print("No Session_Data folder")
        return
        
    sessions = sorted(glob.glob(os.path.join(base_dir, "Session_*")))
    if not sessions:
        print("No sessions found")
        return
        
    latest = sessions[-1]
    analyze_session(latest)

if __name__ == "__main__":
    main()
