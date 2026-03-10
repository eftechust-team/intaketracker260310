import os
import cv2
import numpy as np
import csv
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(
    description=(
        "merge mask for each food item"
    )
)

parser.add_argument(
    "--output",
    type=str,
    default='../data/M1',
    help=(
        "Path to the directory where results will be output. Output will be a folder "
    ),
)

args = parser.parse_args()


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {path}")
    return (mask > 127).astype(np.uint8)  # binary mask 0 or 1

def merge_masks_by_category(mask_folder, csv_txt_path, output_folder, merge_by='category_id'):
    """
    Merge masks grouped by category_id or category_name as per csv_txt_path.
    
    Parameters:
    - mask_folder: folder with mask images, mask files named as {id}.png
    - csv_txt_path: path to csv or txt file with header including columns:
        id, category_id, category_name, ...
    - output_folder: where to save merged masks
    - merge_by: 'category_id' or 'category_name' (default 'category_id')
    
    Returns:
    - dict: {category_key: merged_mask (np.ndarray)}
    """
    category_map = defaultdict(list)
    
    with open(csv_txt_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mask_id = row['id'].strip()
            category_key = row[merge_by].strip()
            category_map[category_key].append(mask_id)
    
    merged_masks = {}
    for category_key, mask_ids in category_map.items():
        merged_mask = None
        for mask_id in mask_ids:
            mask_filename = f"{mask_id}.png"  # adjust extension if needed
            mask_path = os.path.join(mask_folder, mask_filename)
            if not os.path.isfile(mask_path):
                print(f"Warning: mask file {mask_filename} not found, skipping.")
                continue
            mask = load_mask(mask_path)
            if merged_mask is None:
                merged_mask = np.zeros_like(mask, dtype=np.uint8)
            merged_mask = np.logical_or(merged_mask, mask)
        if merged_mask is None:
            print(f"Warning: No masks found for category '{category_key}'")
            continue
        merged_mask = merged_mask.astype(np.uint8)

        os.makedirs(output_folder, exist_ok=True)
        safe_name = category_key.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(output_folder, f"{safe_name}.png")
        cv2.imwrite(save_path, merged_mask * 255)
        print(f"Saved merged mask for '{category_key}' with {len(mask_ids)} masks to: {save_path}")
        merged_masks[category_key] = merged_mask
    
    return merged_masks


if __name__ == "__main__":
    # name_list = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',\
    #              'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']
    
    mask_folder = f"{args.output}/sam_mask"
    csv_txt_path = f"{args.output}/sam_mask_label/semantic_masks_category.txt"  # your CSV-like file with header
    output_folder = f"{args.output}/merged_mask"

    merged_masks = merge_masks_by_category(mask_folder, csv_txt_path, output_folder, merge_by='category_id')