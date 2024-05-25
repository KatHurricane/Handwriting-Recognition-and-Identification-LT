import os
from collections import defaultdict

# Define the Lithuanian alphabet excluding numbers
lithuanian_alphabet = 'AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ'
valid_extensions = ('.png', '.jpg')

def count_first_letters(directory):
    letter_counts = defaultdict(int)
    total_files = 0
    
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(valid_extensions):  # Ensure filename has a valid extension
                first_letter = filename[0].upper()
                if first_letter in lithuanian_alphabet:
                    letter_counts[first_letter] += 1
                else:
                    letter_counts['OTHER'] += 1  # To handle unexpected characters
                total_files += 1
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    
    return letter_counts, total_files

def save_counts_to_file(all_counts, total_file_counts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for letter in lithuanian_alphabet:
            line = f"{letter}:"
            total_count = 0
            for directory, (counts, _) in all_counts.items():
                line += f" {directory}={counts[letter]}"
                total_count += counts[letter]
            line += f" TOTAL={total_count}\n"
            f.write(line)
        if 'OTHER' in all_counts[next(iter(all_counts))][0]:
            line = "OTHER:"
            total_count = 0
            for directory, (counts, _) in all_counts.items():
                line += f" {directory}={counts['OTHER']}"
                total_count += counts['OTHER']
            line += f" TOTAL={total_count}\n"
            f.write(line)
        
        # Write total file counts for each directory and the grand total
        grand_total_files = sum(total_file_counts.values())
        f.write("\nTOTAL FILE COUNTS:\n")
        for directory, count in total_file_counts.items():
            f.write(f"{directory}: {count}\n")
        f.write(f"GRAND TOTAL: {grand_total_files}\n")

# List of directories to scan
directories_to_scan = [
    './output_letters',
    './augmented_images',
    './input_synthetic_data'
]

# Dictionary to store counts for each directory
all_counts = {}
total_file_counts = {}

# Get the counts for each directory
for directory in directories_to_scan:
    counts, total_files = count_first_letters(directory)
    all_counts[directory] = (counts, total_files)
    total_file_counts[directory] = total_files

# Save the counts to a text file
output_file_path = 'letter_counts.txt'
save_counts_to_file(all_counts, total_file_counts, output_file_path)

print(f"Letter counts saved to {output_file_path}")
