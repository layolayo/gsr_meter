import json
import os
import sys

# Add current directory to path so we can import process_runner
sys.path.append(os.getcwd())
from process_runner import ProcessRunner

def test_else_logic():
    print("Running Else Logic Test...")
    
    # We'll use a mock structures since we want to be sure about the behavior
    # This structure mimics "Reciprocal Processing"
    mock_structure = {
        "iterate_assessment": "item",
        "pattern": [
            {
                "pattern": [
                    {"key": "P1", "text": "Q1 [else]for [item]"},
                    {
                        "pattern": [
                            {"key": "C1", "text": "CQ1 [else]for [item]"}
                        ],
                        "repeat": 2
                    }
                ],
                "repeat": 2
            }
        ]
    }
    
    runner = ProcessRunner("processes.json", audio_enabled=False)
    runner.assessments["item"] = ["Faith"]
    
    # Compile
    steps = runner.compile_process(mock_structure)
    
    for i, s in enumerate(steps):
        print(f"[{i+1}] {s['text']}")
    
    # Expected results:
    # [1] Q1 for Faith (First P1)
    # [2] CQ1 for Faith (First C1 of first P1)
    # [3] CQ1 else for Faith (Second C1 of first P1)
    # [4] Q1 else for Faith (Second P1)
    # [5] CQ1 for Faith (First C1 of second P1) <-- WITHOUT ELSE! (This is what we want to fix)
    # [6] CQ1 else for Faith (Second C1 of second P1)
    
    texts = [s['text'] for s in steps]
    
    assert "else" not in texts[0], "Step 1 should not have else"
    assert "else" not in texts[1], "Step 2 should not have else"
    assert "else" in texts[2], "Step 3 should have else"
    assert "else" in texts[3], "Step 4 should have else (P1 repeating)"
    
    # The fix target:
    if "else" in texts[4]:
        print("\nFAIL: Step 5 contains 'else'. Logic is NOT reset for new parent iteration.")
    else:
        print("\nPASS: Step 5 does NOT contain 'else'. Logic IS reset for new parent iteration.")

if __name__ == "__main__":
    test_else_logic()
