from fastai.vision.all import *
import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from natsort import natsorted

def get_im_txt_pths(dataset_dir, im_extensions=('.jpg', '.png', '.tiff')):
    """
    Retrieves image and corresponding YOLO annotation file paths.

    Parameters:
    -----------
    dataset_dir : Path
        Dataset directory where full-size PID images and their annotations in YOLO format are present.
    im_extensions : tuple, optional
        Allowed image file extensions (default: ('.jpg', '.png', '.tiff')).

    Returns:
    --------
    im_pths : list
        List of image file paths sorted naturally.
    txt_pths : list
        List of annotation file paths sorted naturally.
    """
    # Ensure directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    
    # Get image and annotation paths
    im_pths = natsorted(get_files(dataset_dir, extensions=im_extensions))
    txt_pths = natsorted(get_files(dataset_dir, extensions='.txt'))
    
    if len(im_pths) == 0:
        raise ValueError(f"No images found in {im_dir} with extensions {im_extensions}")
    
    if len(txt_pths) == 0:
        raise ValueError(f"No annotation files found in {txt_dir}")
    
    # Ensure the number of images matches the number of annotations
    if len(im_pths) != len(txt_pths):
        raise ValueError(f"Mismatch: {len(im_pths)} images and {len(txt_pths)} annotations found.")
    
    return im_pths, txt_pths




def copy_files_to_directory(file_paths, dest_dir):
    """
    Copy a list of files to a destination directory, creating the directory if it doesn't exist.

    Parameters:
    -----------
    file_paths : list
        List of file paths to be copied.
    dest_dir : str or Path
        Destination directory to copy the files to.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    print(f"Copying {len(file_paths)} files to {dest_dir}...")
    for file_path in map(Path, file_paths):
        dest_file = dest_dir / file_path.name
        shutil.copy(file_path, dest_file)
    
    print(f"Successfully copied {len(file_paths)} files to {dest_dir}")