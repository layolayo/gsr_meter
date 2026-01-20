
import tkinter as tk
from modules.manual_viewer import ManualViewer
import os

def test_manual():
    print("Testing Manual Viewer Integration...")
    
    # 1. Verify File Exists
    calib_path = "manual/calibration.md"
    if not os.path.exists(calib_path):
        print(f"❌ FAIL: {calib_path} not found.")
        return
    print(f"✅ PASS: {calib_path} exists.")
    
    # 2. Verify Config in v40_manual.py (Static Check)
    with open("v40_manual.py", "r") as f:
        content = f.read()
        if '("Device Calibration", "manual/calibration.md", "#e67e22")' in content:
            print("✅ PASS: v40_manual.py configuration appears correct.")
        else:
            print("❌ FAIL: Configuration line not found in v40_manual.py")

    # 3. Test Loading (Headless-ish)
    try:
        root = tk.Tk()
        root.withdraw() # Hide
        
        viewer = ManualViewer(
            pages=[("Device Calibration", "manual/calibration.md", "#e67e22")]
        )
        # Mock UI creation to ensure no crash
        viewer.show()
        
        # Check if text was loaded
        # We need to access the text widget content.
        # viewer.doc_text is the widget.
        text_content = viewer.doc_text.get("1.0", tk.END)
        
        if "Device Calibration Guide" in text_content and "Hubbard" in text_content:
             print("✅ PASS: Content loaded efficiently into Viewer.")
        else:
             print("❌ FAIL: Content matched failed.")
             print(f"snippet: {text_content[:100]}...")
             
        root.destroy()
    except Exception as e:
        print(f"❌ FAIL: Exception during ManualViewer test: {e}")

if __name__ == "__main__":
    test_manual()
