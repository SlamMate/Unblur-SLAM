#!/usr/bin/env python3
"""
Extract sharp frame subsets from index files based on specified indices.
"""

import os

# Define the sharp frame indices for each dataset
SHARP_FRAME_INDICES = {
    'TUM/fr1_desk': [13, 20, 21, 35, 44, 60, 61, 65, 67, 72, 82, 85, 87, 88],
    'TUM/fr2_xyz': list(range(42)),  # [0, 1, 2, ..., 41]
    'TUM/fr3_office': [0, 1, 3, 11, 14, 16, 18, 22, 23, 24, 25, 31, 33, 34, 44, 46, 47, 50, 
                       51, 52, 66, 69, 70, 73, 75, 77, 82, 83, 84, 85, 86, 87, 88, 94, 95, 96, 
                       97, 100, 101, 102, 103, 104, 105, 109, 110, 111, 112, 113, 123, 124, 125, 
                       137, 145, 147, 149, 151, 152]
}

# Define file paths
BASE_PATH = './scripts'
FILES = {
    'TUM/fr1_desk': os.path.join(BASE_PATH, 'fr1_desk_indices.txt'),
    'TUM/fr2_xyz': os.path.join(BASE_PATH, 'fr2_xyz_indices.txt'),
    'TUM/fr3_office': os.path.join(BASE_PATH, 'fr3_office_indices.txt')
}

def extract_sharp_frames(input_file, output_file, indices):
    """
    Extract lines at specified indices from input file and save to output file.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        indices: List of line indices to extract (0-based)
    """
    try:
        # Read all lines from input file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Extract lines at specified indices
        sharp_lines = []
        for idx in indices:
            if idx < len(lines):
                sharp_lines.append(lines[idx])
            else:
                print(f"Warning: Index {idx} is out of range for {input_file} (has {len(lines)} lines)")
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.writelines(sharp_lines)
        
        print(f"✓ Extracted {len(sharp_lines)} sharp frames from {os.path.basename(input_file)}")
        print(f"  Saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File not found - {input_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    print("Extracting sharp frame subsets...\n")
    
    for dataset, input_file in FILES.items():
        # Get the sharp frame indices for this dataset
        indices = SHARP_FRAME_INDICES[dataset]
        
        # Create output filename (add '_sharp' suffix)
        base_name = os.path.basename(input_file)
        name_parts = base_name.rsplit('.', 1)
        if len(name_parts) == 2:
            output_name = f"{name_parts[0]}_sharp.{name_parts[1]}"
        else:
            output_name = f"{base_name}_sharp"
        
        output_file = os.path.join(os.path.dirname(input_file), output_name)
        
        # Extract sharp frames
        print(f"Processing {dataset}:")
        print(f"  Input: {input_file}")
        print(f"  Extracting {len(indices)} indices")
        extract_sharp_frames(input_file, output_file, indices)
        print()

if __name__ == "__main__":
    main()