import numpy as np
import re

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 
            deletions = current_row[j] + 1       
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def find_closest_pattern_match(user_input):
    # Examples and hints without spaces and in English
    samples_and_hints = [
        ("s1=-12.45", "Format: s<number> = <number> (integer or decimal, can be negative)"),
        ("s1=3.45", "Format: s<number> = <number> (integer or decimal, can be negative)"),
        ("s1=4", "Format: s<number> = <number> (integer or decimal, can be negative)"),
        ("s1=dot(v2,v3)", "Format: s<number> = dot(v<number>, v<number>) (dot product operation of two vectors)"),
        ("s1=s2-s3", "Format: s<number> = s<number> - s<number> (subtraction of one s variable from another)"),
        ("s1=s2*s3", "Format: s<number> = s<number> * s<number> (multiplication of one s variable by another)"),
        ("v1=s1*v2", "Format: v<number> = s<number> * v<number> (multiplication of a vector by a scalar)"),
        ("v=s1*v2", "Format: v<number> = s<number> * v<number> (multiplication of a vector by a scalar)"),
        ("v1=s*v2", "Format: v<number> = s<number> * v<number> (multiplication of a vector by a scalar)"),
        ("v1=s1*v", "Format: v<number> = s<number> * v<number> (multiplication of a vector by a scalar)"),
        ("v1=s*v", "Format: v<number> = s<number> * v<number> (multiplication of a vector by a scalar)"),
        ("v=s*v2", "Format: v<number> = s<number> * v<number> (multiplication of a vector by a scalar)"),
        ("v=s*v", "Format: v<number> = s<number> * v<number> (multiplication of a vector by a scalar)"),
        ("v1=v2+v3", "Format: v<number> = v<number> + v<number> (addition of two vectors)")
    ]

    # Remove spaces from user input for comparison
    user_input_no_spaces = user_input.replace(" ", "")
    best_match = None
    best_distance = np.inf
    best_hint = ""
    for sample, hint in samples_and_hints:
        distance = levenshtein_distance(user_input_no_spaces, sample)
        if distance < best_distance:
            best_distance = distance
            best_match = sample
            best_hint = hint

    match_percentage = (1 - best_distance / max(len(user_input_no_spaces), len(best_match))) * 100
    return best_match, match_percentage, best_hint

def check_input(user_input):
    patterns = [
        r's\d+\s*=\s*-?\d+(\.\d+)?',
        r's\d+\s*=\s*dot\(v\d+\s*,\s*v\d+\)',
        r's\d+\s*=\s*s\d+\s*-\s*s\d+',
        r's\d+\s*=\s*s\d+\s*\*\s*s\d+',
        r'v\d+\s*=\s*s\d+\s*\*\s*v\d+',
        r'v\d+\s*=\s*v\d+\s*\+\s*v\d+'
    ]

    for pattern in patterns:
        if re.match(pattern, user_input):
            return True
    return False

categories = ["Setup", "Predict", "Learn"]

def enter_alg(MAX_OP = 1000):
    alg = "def Setup():"

    current_category_idx = 0
    current_op = 0
    while current_category_idx <= 2:
        if current_category_idx == 2 and current_op == MAX_OP:
            return alg
        user_input = ""
        if (current_op > 0 and current_category_idx < 2):
            user_input = input(f"Enter operation for {categories[current_category_idx]}, or Enter {categories[current_category_idx + 1]} to start enter for {categories[current_category_idx + 1]}: ")
        elif (current_op > 0 and current_category_idx == 2):
            user_input = input(f"Enter operation for {categories[current_category_idx]}, or Stop: ")
        else:
            user_input = input(f"Enter operation for {categories[current_category_idx]}: ")
        
        if current_category_idx < 2 and (categories[current_category_idx + 1] == user_input or current_op == MAX_OP):
            current_category_idx += 1
            current_op = 0
            alg += "\n" + f"def {categories[current_category_idx]}():"
            continue
        
        if current_category_idx == 2 and current_op > 0 and "Stop" == user_input:
            return alg

        if check_input(user_input):
            alg += "\n\t" + user_input
            current_op += 1
            continue
        else:
            best_match, match_percentage, best_hint = find_closest_pattern_match(user_input)
            if match_percentage > 10:
                print(f"Wrong input. Suggestion: {best_hint}")
            else:
                print(f"Wrong input")
    return alg