import json

FILE = "processes.json"

def validate():
    try:
        with open(FILE, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    library = data.get("question_library", {})
    processes = data.get("processes", [])
    
    missing_keys = set()
    
    print(f"Library has {len(library)} questions.")
    
    # helper to check key
    def check_key(k, context=""):
        if k not in library:
            print(f"MISSING: '{k}' in {context}")
            missing_keys.add(k)
        # else:
            # print(f"OK: {k}")

    # 1. Check Processes
    for p in processes:
        p_name = p.get("name", "Unknown")
        print(f"Checking {p_name}...")
        
        # Check Structure (Pattern)
        if "structure" in p:
            pattern = p["structure"].get("pattern", [])
            for k in pattern:
                check_key(k, context=f"Process '{p_name}' (Pattern)")
                
        # Check Steps (Mixed)
        if "steps" in p:
            for step in p["steps"]:
                if isinstance(step, str):
                    check_key(step, context=f"Process '{p_name}' (Step)")
                elif isinstance(step, dict):
                    # Check Inline Pattern
                    if "pattern" in step:
                        def check_recursive(item, ctx):
                            if isinstance(item, str):
                                check_key(item, context=ctx)
                            elif isinstance(item, list):
                                for sub in item:
                                    check_recursive(sub, ctx)
                            elif isinstance(item, dict):
                                # If the item is a dict inside a pattern list, it might be a nested step definition
                                # We should check if it has a "pattern" or just extract keys?
                                # Based on "processes.json" structure (e.g. repeat blocks), 
                                # the dict likely has keys like "pattern", "repeat"
                                if "pattern" in item:
                                    check_recursive(item["pattern"], ctx)
                                # If it has steps?
                                if "steps" in item:
                                    check_recursive(item["steps"], ctx)

                        check_recursive(step["pattern"], f"Process '{p_name}' (Inline Pattern)")
                            
    # 2. Check Closing Questions
    closing = data.get("closing_questions", [])
    print("Checking Closing Questions...")
    
    # Closing can be list of objs or obj with steps
    # Normalize to list of steps if needed
    steps_to_check = []
    
    if isinstance(closing, list):
         # Could be list of objects or list of steps (if the code supported it, but closing is usually list of objects)
         # In the updated format user might have put { "steps": [...] } in the list?
         # Let's inspect what user did in Step 435: keys in closing steps.
         for item in closing:
             if isinstance(item, dict) and "steps" in item:
                 steps_to_check.extend(item["steps"])
             elif isinstance(item, str):
                 steps_to_check.append(item)
    elif isinstance(closing, dict) and "steps" in closing:
         steps_to_check.extend(closing["steps"])
         
    for s in steps_to_check:
        if isinstance(s, str):
            check_key(s, context="Closing Questions")
            
    if not missing_keys:
        print("\nSUCCESS: All process steps reference valid questions.")
    else:
        print(f"\nFAILURE: Found {len(missing_keys)} missing keys.")

if __name__ == "__main__":
    validate()
