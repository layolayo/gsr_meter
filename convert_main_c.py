
import re
import json
import os

MAIN_C_PATH = "/home/matthew/PycharmProjects/gsr_meter/main.c"
OUTPUT_JSON_PATH = "/home/matthew/PycharmProjects/gsr_meter/holigral_processes.json"

question_library = {}
processes = []

# Helper function definitions (hardcoded based on main.c analysis)
# We map these function names to a list of step structures
helpers = {
    "step_forward": { "repeat": 6, "steps": ["fwd_sfa", "fwd_sfb"] },
    "inside_questions": { "repeat": 6, "steps": ["inside_isa"] },
    "saturday_questions": { "steps": ["saturday_saa"] },
    "sunday_questions": { "steps": ["sunday_su1", {"pattern": ["sunday_su2"], "repeat": 5}, "sunday_su7"] },
    "monday_questions": { "steps": ["monday_mo1", {"pattern": ["monday_mo2"], "repeat": 5}, "monday_mo7"] },
    "tuesday_questions": { "steps": [{"pattern": ["tuesday_tu2"], "repeat": 6}, "tuesday_tu7"] },
    "wednesday_questions": { "steps": [{"pattern": ["wednesday_wed2"], "repeat": 6}, "wednesday_wed7"] },
    "thursday_questions": { "steps": [{"pattern": ["thursday_th2"], "repeat": 6}, "thursday_th7"] },
    "friday_questions": { "steps": [{"pattern": ["friday_f1"], "repeat": 6}] },
}

# Pre-populate library with helper questions (extracted manually or we can let the parser find them if they use interact)
# Actually, the parser will find the interact calls inside these functions if we parse them too.
# But providing the structure is easier if we treat them as known blocks.
# I'll let the parser find the questions for helpers too by parsing the helper functions first.

def parse_string(s):
    # Remove C string quotes and newlines/formatting
    return s.strip('"').replace('\\n', ' ').strip()


def parse_main_c():
    with open(MAIN_C_PATH, 'r') as f:
        lines = f.readlines()

    current_function = None
    function_steps = {} # func_name -> list of steps
    local_vars = {} # func_name -> {var_name: value}
    
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        
        # Function definition
        m_func = re.match(r'void\s+(\w+)\s*\(', line)
        if m_func:
            current_function = m_func.group(1)
            function_steps[current_function] = []
            local_vars[current_function] = {}
            line_idx += 1
            continue

        if not current_function:
            line_idx += 1
            continue

        # Check for strcpy(var, "string")
        m_strcpy = re.match(r'strcpy\s*\(\s*([^,]+)\s*,\s*"(.*)"\s*\);', line)
        if m_strcpy:
            var_name = m_strcpy.group(1).strip()
            # Handle potential escaped quotes in C string
            val = m_strcpy.group(2).replace('\\"', '"').replace('\\n', ' ')
            local_vars[current_function][var_name] = val
            line_idx += 1
            continue
            
        # Check for sprintf(var, "string") - simplified
        m_sprintf = re.match(r'sprintf\s*\(\s*([^,]+)\s*,\s*"(.*)"\s*\);', line)
        if m_sprintf:
             var_name = m_sprintf.group(1).strip()
             val = m_sprintf.group(2).replace('\\"', '"').replace('\\n', ' ')
             local_vars[current_function][var_name] = val
             line_idx += 1
             continue

        # Check for loop start
        m_for = re.match(r'for\s*\(\s*i\s*=\s*0\s*;\s*i\s*<\s*(\d+)\s*;', line)
        if m_for:
            count = int(m_for.group(1))
            block_steps = []
            
            # Simple block extraction (assumes well-formatted code)
            if '{' in line or (line_idx+1 < len(lines) and '{' in lines[line_idx+1]):
                if '{' not in line: line_idx += 1
                line_idx += 1
                nest_level = 1
                while line_idx < len(lines) and nest_level > 0:
                    sub_line = lines[line_idx].strip()
                    if '{' in sub_line: nest_level += 1
                    if '}' in sub_line: nest_level -= 1
                    if nest_level == 0: break
                    
                    parse_line_content(sub_line, block_steps, local_vars.get(current_function, {}))
                    line_idx += 1
            else:
                line_idx += 1
                sub_line = lines[line_idx].strip()
                parse_line_content(sub_line, block_steps, local_vars.get(current_function, {}))
                
            function_steps[current_function].append({
                "pattern": block_steps,
                "repeat": count
            })
            line_idx += 1
            continue

        # Handle interact
        parse_line_content(line, function_steps[current_function], local_vars.get(current_function, {}))
        
        # Handle end of function
        if line == '} // end of ' + current_function or (line == '}' and len(lines) > line_idx+1 and 'void' in lines[line_idx+1]):
             current_function = None

        line_idx += 1
        
    return function_steps

def parse_line_content(line, steps_list, current_locals):
    # interact("question", "id", "step")
    m_interact = re.match(r'interact\s*\((.*)\);', line)
    if m_interact:
        args_str = m_interact.group(1)
        # Split by comma, respecting quotes is hard with regex split alone
        # But here valid args are either "literal" or variable
        # We can try a simple split and reconstruct if needed, but given the file format...
        
        # Let's split by last two commas
        parts = args_str.rsplit(',', 2)
        if len(parts) == 3:
            q_arg = parts[0].strip()
            proc_id = parse_string(parts[1])
            step_id = parse_string(parts[2])
            
            # Resolve q_arg
            q_text = ""
            if q_arg.startswith('"'):
                q_text = parse_string(q_arg)
            elif q_arg in current_locals:
                q_text = current_locals[q_arg]
            else:
                q_text = q_arg # fallback
            
            key = f"{proc_id}_{step_id}"
            question_library[key] = {"text": q_text}
            steps_list.append(key)
        return

    for helper_name in helpers:
        if line.startswith(f"{helper_name}("):
            h = helpers[helper_name]
            if "repeat" in h and "pattern" not in h:
                 steps_list.append({"pattern": h["steps"], "repeat": h["repeat"]})
            elif "steps" in h:
                 steps_list.extend(h["steps"])
            return

parsed_funcs = parse_main_c()

# Now construct the processes list from the top-level processes
# We filter for processes we care about (the ones in setupprocessnames + distance etc)
# Or just dump all parsed functions that have steps?

target_processes = [
    "ancestors", "relating", "re_scaling", "distance_ib", "exorcism", 
    "nonlexic_pronoun", "projections", "pronoun", "simple_pronoun", "turning",
    "betweens", "critic_busting", "freudian_defenses", "installing", "self_defence",
    "the_code", "clean_space", "deinterleaving", "marketing", "metaphor",
    "metaphor_transfer", "perceptual_space", "story_analysis", "story_busting",
    "story_insight", "clean_worlds", "gods", "pulling_back", "pulling_back2",
    "rock_solid", "truth", "ventriloquism", "boolean", "business_science",
    "emerging_moving", "group_forms", "habit_busting", "inging", "scenarios",
    "clean_shamanism", "emergent_poetry", "no_sacrifices", "relationship",
    "releasing", "sigint", "job_profile", "learning", "cultural", "retreat"
]

for pname in target_processes:
    if pname in parsed_funcs and len(parsed_funcs[pname]) > 0:
        processes.append({
            "name": pname.replace('_', ' ').title(),
            "steps": parsed_funcs[pname]
        })

output_data = {
    "question_library": question_library,
    "processes": processes
}

with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Converted {len(processes)} processes with {len(question_library)} questions.")
