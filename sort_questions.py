import json

FILE = "processes.json"

try:
    with open(FILE, 'r') as f:
        data = json.load(f)

    library = data.get("question_library", {})
    
    # Sort dictionary items by value['text']
    # We use a lambda that handles case-insensitive sorting
    sorted_items = sorted(library.items(), key=lambda item: item[1]['text'].lower())
    
    # Reconstruct dictionary
    sorted_library = {k: v for k, v in sorted_items}
    
    data["question_library"] = sorted_library
    
    with open(FILE, 'w') as f:
        # custom dump to preserve order since Python 3.7+ preserves insertion order
        json.dump(data, f, indent=4)
        
    print(f"Sorted {len(sorted_library)} questions by text.")

except Exception as e:
    print(f"Error: {e}")
