
def process_4():
    # Outer loop: repeat 6 times
    for _ in range(6):
        print("And, where are you now?")
        input()  # Wait for Enter
        
        # Inner loop: repeat 3 times
        for i in range(3):
            if i == 0:
                print("And, where have you been?")
            else:
                print("And, where else have you been?")
            input()  # Wait for Enter
            
            print("And, compare that to where you are now.")
            input()  # Wait for Enter
            
            if i == 0:
                print("And, where might you be?")
            else:
                print("And, where else might you be?")
            input()  # Wait for Enter
            
            print("And, compare that to where you are now.")
            input()  # Wait for Enter
    
    # Final question
    print("And, where are you now?")
    input()  # Wait for Enter

if __name__ == "__main__":
    process_4()
