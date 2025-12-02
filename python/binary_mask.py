import os
import argparse
import time
import numpy as np
from preprocessing_utils import *
from visual_utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate binary mask for biological images to filter out background.")
    parser.add_argument("path", type=str, help="Path to the directory containing TIFF images.")
    parser.add_argument("--randomly_sampled", action="store_true", help="If set, randomly sample images (load_partial_tiff_series). If not, load all images (load_tiff_series). Recommended for large datasets.")
    parser.add_argument("--brightness_gradient", action="store_true", help="If set, apply a brightness gradient correction")
    parser.add_argument("--tiff_stack", action='store_true', help="Specify if the input is a single 3D TIFF file instead of a directory of 2D TIFF files. Note: an entire directory must be handed and this should be the only .tiff file in there.")

    args = parser.parse_args()
    
    if not os.path.isdir(args.path):
        raise ValueError(f"The specified path {args.path} is not a valid directory.")
    
    print("Loading TIFF series...")
    start_time = time.time()
    if args.randomly_sampled == 1:
        image_series = load_partial_tiff_series(args.path)
    elif args.tiff_stack:
        image_series = load_tiff_3d(os.path.join(args.path, [f for f in os.listdir(args.path) if f.endswith('.tiff') or f.endswith('.tif')][0]))
    else:
        image_series = load_tiff_series(args.path)
    elapsed = time.time() - start_time
    print(f"Loaded {image_series.shape[0]} frames of size {image_series.shape[1]}x{image_series.shape[2]}.")
    print(f"Loading took {elapsed:.2f} seconds.")

    threshold, invert = visualize_binary_mask(image_series, args.brightness_gradient)
    mask = binary_mask(image_series, threshold, invert, args.brightness_gradient)
    invert_str = "_inv" if invert else ""
    filename = f"binary_mask_thresh{threshold}{invert_str}grad_{args.brightness_gradient}.npy"
    np.save(os.path.join(args.path, filename), mask)

# Example usage: python binary_mask.py /Users/josephbeller/Library/CloudStorage/Box-Box/Larina_team_folder/Joesph/Endometriosis_Organoids_data/070825die63p9_LSM02DC_4p4ms_2000x3000_0p5Vx0V_10ms_225khz_a/images