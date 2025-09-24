#!/usr/bin/env python3
"""
Rearrange CSV files according to the image number extracted from the filename.
"""

import csv
import re
from pathlib import Path

def extract_image_number(image_name):
    """
    Extract the numeric part from image filename (e.g., '718.png' -> 718).
    
    Args:
        image_name: Image filename like '718.png'
    
    Returns:
        int: The numeric part of the filename
    """
    # Extract number from filename using regex
    match = re.search(r'(\d+)\.png', image_name)
    if match:
        return int(match.group(1))
    else:
        # Fallback: try to extract any number from the string
        numbers = re.findall(r'\d+', image_name)
        if numbers:
            return int(numbers[0])
        else:
            return 0  # Default value for sorting

def rearrange_csv_by_image_number(input_csv, output_csv):
    """
    Rearrange CSV file by sorting rows according to image number.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
    """
    print(f"🔄 Rearranging {input_csv}...")
    
    # Read the CSV file
    rows = []
    with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            rows.append(row)
    
    print(f"   📊 Read {len(rows)} rows")
    
    # Sort by image number
    rows_sorted = sorted(rows, key=lambda x: extract_image_number(x['image_name']))
    
    # Write the sorted CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    
    print(f"   ✅ Sorted and saved to {output_csv}")
    
    # Show some statistics
    if rows_sorted:
        first_image = extract_image_number(rows_sorted[0]['image_name'])
        last_image = extract_image_number(rows_sorted[-1]['image_name'])
        print(f"   📈 Image number range: {first_image} to {last_image}")
        
        # Show first few and last few entries
        print(f"   🔢 First 3 images: {[extract_image_number(row['image_name']) for row in rows_sorted[:3]]}")
        print(f"   🔢 Last 3 images: {[extract_image_number(row['image_name']) for row in rows_sorted[-3:]]}")

def main():
    """Main function to rearrange CSV files."""
    print("🔄 Rearranging CSV files by image number...")
    print("=" * 60)
    
    # Define input and output files
    files_to_process = [
        {
            'input': 'subsampling/Subsets/training_data_percent_split.csv',
            'output': 'subsampling/Subsets/training_data_percent_split_sorted.csv'
        },
        {
            'input': 'subsampling/Subsets/training_data_pixel_count.csv',
            'output': 'subsampling/Subsets/training_data_pixel_count_sorted.csv'
        }
    ]
    
    # Process each file
    for file_info in files_to_process:
        input_path = Path(file_info['input'])
        output_path = Path(file_info['output'])
        
        if input_path.exists():
            rearrange_csv_by_image_number(input_path, output_path)
            print()
        else:
            print(f"❌ File not found: {input_path}")
            print()
    
    print("🎉 All CSV files have been rearranged by image number!")

if __name__ == "__main__":
    main()
