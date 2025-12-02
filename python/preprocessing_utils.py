import numpy as np
import os
from PIL import Image
from natsort import natsorted
import imageio.v2 as imageio
import time
import random
from tqdm import tqdm
import scipy

def apply_brightness_gradient_correction(img):
    """
    Applies a brightness gradient correction.
    Args:
        img (numpy array): A 2D numpy array representing the image.
    Returns:
        numpy array: The corrected image with brightness gradient applied.
    """
    gradient_vec = np.linspace(np.percentile(img, 50), np.min(img), img.shape[0]).reshape(-1,1)
    gradient_mat = np.repeat(gradient_vec, img.shape[1], axis = 1)
    img = np.clip(img - gradient_mat, 0, None)
    img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    return img

def binary_mask(image_series, threshold, invert, brightness_gradient=False):
    """
    Generate a binary mask from the image series based on the specified threshold and invert flag.
    
    Args:
        image_series (numpy array): A 3D numpy array of shape (T, H, W) representing the image series.
        threshold (int): The threshold value for generating the binary mask.
        invert (bool): If True, inverts the mask.
        
    Returns:
        numpy array: A binary mask of shape (H, W).
    """
    avg_img = np.mean(image_series, axis=0)
    if brightness_gradient:
        avg_img = apply_brightness_gradient_correction(avg_img)
    if invert:
        mask = avg_img < threshold
    else:
        mask = avg_img >= threshold
    return mask.astype(np.uint8)

def get_frame_rate(images_path):
    """
    Get the frame rate from the directory name based on the "ms" value. Uses the ms value that appears left-most in the string.
    """
    dir_name = os.path.basename(os.path.dirname(os.path.normpath(images_path)))
    parts = dir_name.split("_")
    ms_value = None
    # Iterate in reverse to get the last occurrence of a part ending with "ms"
    for part in reversed(parts):
        if part.endswith("ms"):
            ms_str = part.replace("ms", "").replace("p", ".")
            try:
                ms_value = float(ms_str)
                break
            except ValueError:
                continue

    if ms_value is None or ms_value == 0:
        raise ValueError("Could not extract a valid ms value from directory name.")

    frame_rate = 1 / (ms_value / 1000)
    return frame_rate

def load_binary_mask(mask_path):
    """
    Load a binary mask from the specified path and apply it to the images in the given directory.
    Args:
        mask_path (str): Path to the binary mask file, which ideally should be .npy (I do not know of other supported formats).
    Returns:
        numpy array: A binary mask of shape (H, W).
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"The specified mask file {mask_path} does not exist.")
    
    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise ValueError("The loaded mask must be a 2D array (H, W). If this is for 4D DyCOCT, look for functions to load in 3D binary mask.")
    
    return mask

def load_partial_tiff_series(images_path):
    """
    Load up to 200 random TIFF images from the specified directory.
    If there are 200 or fewer, load all of them.
    Args:
        images_path (str): Path to the directory containing TIFF images.
    Returns:
        numpy array: A 3D numpy array of shape (T, H, W)
                     where T is the number of frames, H is the height, and W is the width
                     of the images.
    NOTE: This function is used to load a random subset of images 
    for preprocessing steps such as visualizing the binary
    mask, which saves time for large datasets.
    """
    filenames = natsorted([f for f in os.listdir(images_path) if f.endswith(".tiff") or f.endswith(".tif")])
    if len(filenames) > 200:
        filenames = random.sample(filenames, 50)
        filenames = natsorted(filenames)  # Optional: sort for consistent order
    images = []
    for fname in tqdm(filenames, desc="Loading partial TIFF images", unit="image"):
        frame = imageio.imread(os.path.join(images_path, fname))
        images.append(frame)
    image = np.stack(images, axis=0)
    return image

def load_tiff_series(images_path):
    """
    Load all TIFF images from the specified directory with a progress bar.
    Args:
        images_path (str): Path to the directory containing TIFF images.
    Returns:
        numpy array: A 3D numpy array of shape (T, H, W)
                     where T is the number of frames, H is the height, and W is the width
                     of the images.
    """
    filenames = natsorted([f for f in os.listdir(images_path) if f.endswith(".tiff") or f.endswith(".tif")])
    images = []
    for fname in tqdm(filenames, desc="Loading TIFF images", unit="image"):
        frame = imageio.imread(os.path.join(images_path, fname))
        images.append(frame)
    image = np.stack(images, axis=0)
    return image

def load_tiff_3d(tiff_path):
    """
    Load a single multi-frame TIFF file (3D TIFF stack).
    Args:
        tiff_path (str): Path to the multi-frame TIFF file.
    Returns:
        numpy array: A 3D numpy array of shape (T, H, W)
                     where T is the number of frames, H is the height, and W is the width
                     of the images.
    """
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"The specified TIFF file {tiff_path} does not exist.")
    
    # Load the multi-frame TIFF
    image_stack = imageio.volread(tiff_path)
    
    # Ensure it's a 3D array (T, H, W)
    if image_stack.ndim == 2:
        # Single frame, add time dimension
        image_stack = np.expand_dims(image_stack, axis=0)
    elif image_stack.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {image_stack.ndim}D array")
    
    print(f"Loaded 3D TIFF with shape: {image_stack.shape}")
    return image_stack