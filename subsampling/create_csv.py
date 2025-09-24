#!/usr/bin/env python3
"""
Parse LoveDa training data and create CSV with image paths, mask paths, and pixel counts.
"""

import os
import csv
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def count_pixels_in_mask(mask_path, forest_class_id=6):
    """
    Count total pixels, forest pixels, and background pixels in a mask.
    
    Args:
        mask_path: Path to the mask image
        forest_class_id: Class ID for forest pixels (default: 6 for LoveDA)
    
    Returns:
        tuple: (total_pixels, forest_pixels, background_pixels)
    """
    try:
        # Load mask image
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        # Count pixels
        total_pixels = mask_array.size
        forest_pixels = np.sum(mask_array == forest_class_id)
        background_pixels = total_pixels - forest_pixels
        
        return total_pixels, forest_pixels, background_pixels
    
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return 0, 0, 0

def parse_loveda_training_data(root_dir="LoveDa/Train", output_csv="training_data_analysis.csv"):
    """
    Parse LoveDa training data and create CSV with pixel analysis.
    
    Args:
        root_dir: Root directory containing Train folder
        output_csv: Output CSV file path
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Error: Directory {root_dir} does not exist!")
        return
    
    # Prepare CSV data
    csv_data = []
    
    # Process both Rural and Urban scenes
    for scene in ["Rural", "Urban"]:
        scene_path = root_path / scene
        
        if not scene_path.exists():
            print(f"Warning: Scene directory {scene} does not exist!")
            continue
        
        images_dir = scene_path / "images_png"
        masks_dir = scene_path / "masks_png"
        
        if not images_dir.exists() or not masks_dir.exists():
            print(f"Warning: Missing images_png or masks_png in {scene}")
            continue
        
        print(f"Processing {scene} scene...")
        
        # Get all image files
        image_files = list(images_dir.glob("*.png"))
        
        for image_path in tqdm(image_files, desc=f"Processing {scene}"):
            # Construct corresponding mask path
            mask_path = masks_dir / image_path.name
            
            if not mask_path.exists():
                print(f"Warning: No corresponding mask for {image_path.name}")
                continue
            
            # Count pixels in mask
            total_pixels, forest_pixels, background_pixels = count_pixels_in_mask(mask_path)
            
            # Add to CSV data
            csv_data.append({
                'scene': scene,
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'image_name': image_path.name,
                'total_pixels': total_pixels,
                'forest_pixels': forest_pixels,
                'background_pixels': background_pixels,
                'forest_percentage': (forest_pixels / total_pixels * 100) if total_pixels > 0 else 0,
                'background_percentage': (background_pixels / total_pixels * 100) if total_pixels > 0 else 0
            })
    
    # Write to CSV
    if csv_data:
        fieldnames = [
            'scene', 'image_path', 'mask_path', 'image_name',
            'total_pixels', 'forest_pixels', 'background_pixels',
            'forest_percentage', 'background_percentage'
        ]
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\n✅ CSV created successfully: {output_csv}")
        print(f"📊 Total images processed: {len(csv_data)}")
        
        # Print summary statistics
        total_forest_pixels = sum(row['forest_pixels'] for row in csv_data)
        total_background_pixels = sum(row['background_pixels'] for row in csv_data)
        total_all_pixels = total_forest_pixels + total_background_pixels
        
        print(f"\n📈 Summary Statistics:")
        print(f"   Total Forest Pixels: {total_forest_pixels:,}")
        print(f"   Total Background Pixels: {total_background_pixels:,}")
        print(f"   Total Pixels: {total_all_pixels:,}")
        print(f"   Forest Percentage: {total_forest_pixels/total_all_pixels*100:.2f}%")
        print(f"   Background Percentage: {total_background_pixels/total_all_pixels*100:.2f}%")
        
        # Scene-wise statistics
        rural_data = [row for row in csv_data if row['scene'] == 'Rural']
        urban_data = [row for row in csv_data if row['scene'] == 'Urban']
        
        if rural_data:
            rural_forest = sum(row['forest_pixels'] for row in rural_data)
            rural_total = sum(row['total_pixels'] for row in rural_data)
            print(f"\n🌾 Rural Scene:")
            print(f"   Images: {len(rural_data)}")
            print(f"   Forest Pixels: {rural_forest:,}")
            print(f"   Forest Percentage: {rural_forest/rural_total*100:.2f}%")
        
        if urban_data:
            urban_forest = sum(row['forest_pixels'] for row in urban_data)
            urban_total = sum(row['total_pixels'] for row in urban_data)
            print(f"\n🏙️ Urban Scene:")
            print(f"   Images: {len(urban_data)}")
            print(f"   Forest Pixels: {urban_forest:,}")
            print(f"   Forest Percentage: {urban_forest/urban_total*100:.2f}%")
    
    else:
        print("❌ No data found to write to CSV!")

def main():
    """Main function to run the parsing."""
    print("🔍 Parsing LoveDa Training Data...")
    print("=" * 50)
    
    # Parse training data
    parse_loveda_training_data(
        root_dir="LoveDa/Train/Train",
        output_csv="training_data_analysis.csv"
    )

if __name__ == "__main__":
    main()
