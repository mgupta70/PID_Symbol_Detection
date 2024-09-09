from fastai.vision.all import *
import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from natsort import natsorted
from utils.helpers import *


def class_aware_to_class_agnostic(dataset_dir, folder_name):
    """
    Converts class-aware YOLO annotations to class-agnostic format (single class).
    Copies images and updates annotation files accordingly.

    Parameters:
    -----------
    dataset_dir : str or Path
        Path to the dataset directory containing images and YOLO format annotations.

    Returns:
    --------
    None
    """
    
    dataset_dir = Path(dataset_dir)
    image_paths, annotation_paths = get_im_txt_pths(dataset_dir)  # Retrieve image and annotation file paths

    # Destination directory for the class-agnostic patches
    dest_dir = Path(f"{dataset_dir.parent}/{folder_name}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy images to destination directory
    copy_files_to_directory(image_paths, dest_dir)

    # Step 2: Update annotations to class-agnostic format and save in destination
    print("Converting annotations to class-agnostic format...")
    for annotation_file in annotation_paths:
        with open(annotation_file, 'r') as file:
            lines = file.readlines()

        # Rewrite each annotation with a single class (class 0)
        new_annotation_path = dest_dir / annotation_file.name
        with open(new_annotation_path, 'w') as new_file:
            for line in lines:
                components = line.split()
                bbox = components[1:]  # Extract bbox coordinates
                new_file.write(f"0 {' '.join(bbox)}\n")  # Replace class with 0

    print(f"Class-agnostic conversion complete. Annotations saved to {dest_dir}")

def make_patches_w_overlap(dataset_dir, overlap=0.25, sz=1024, class_aware_folder = 'patches_class_aware', class_agn_folder = 'patches_class_agnostic', n_random_patches = 0):
    """
    Create overlapping patches of images and corresponding YOLO annotations.

    Parameters:
    -----------
    dataset_dir : Path
        Dataset directory where full-size PID images and their annotations in YOLO format are present.
    dest_dir : str
        Destination directory where cropped image patches and updated annotations will be saved.
    overlap : float, optional
        Fractional overlap between patches, default is 0.25 (i.e., 25% overlap).
    sz : int, optional
        Size of the patch (height and width), default is 1024.

    Returns:
    --------
    None
    """

    im_pths, txt_pths = get_im_txt_pths(dataset_dir)
    dest_dir = Path(f'{dataset_dir.parent}/{class_aware_folder}')
    class_aware_dir = dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    
    # Process each image and corresponding annotation file
    for im_pth, txt_pth in zip(im_pths, txt_pths):
        image = cv2.imread(str(im_pth))
        if image is None:
            print(f"Error reading image: {im_pth}")
            continue
        
        H, W = image.shape[:2]

        # Read annotations
        try:
            with open(txt_pth, 'r') as f:
                lines = [list(map(float, line.split())) for line in f.readlines()]
        except Exception as e:
            print(f"Error reading annotation file: {txt_pth} - {e}")
            continue

        arr = np.array(lines)
        class_labels = arr[:, 0].tolist()  # First column: class IDs
        bboxes = arr[:, 1:]               # Remaining columns: YOLO bounding boxes
        
        # Loop over the image grid to create patches with overlap
        step_size = int(sz * (1 - overlap))
        for i in range(0, W, step_size):
            for j in range(0, H, step_size):
                x_min = i
                y_min = j
                x_max = min(W, i + sz)
                y_max = min(H, j + sz)
                
                transform = A.Compose([
                    A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    A.Resize(sz, sz)
                ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5, label_fields=['class_labels']))

                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']

                # Save the cropped image
                cropped_img_name = f"{im_pth.stem}_{i}_{j}.jpg"
                cv2.imwrite(f"{dest_dir}/{cropped_img_name}", transformed_image)

                # Save the corresponding YOLO annotation
                annotation_name = f"{txt_pth.stem}_{i}_{j}.txt"
                with open(f"{dest_dir}/{annotation_name}", 'w') as g:
                    for box, label in zip(transformed_bboxes, transformed_class_labels):
                        x_c, y_c, w, h = box
                        g.write(f"{int(label)} {x_c} {y_c} {w} {h}\n")
        
        #print(f"Processed patches for image: {im_pth.name}")
        if n_random_patches>0:
            make_random_patches_per_sheet(dataset_dir, folder_name=class_aware_folder, sz=sz, n=n_random_patches)
    
    print('********* Class-Aware patches created successfully *************')

    # Create same patches but with Class Agnostic labels
    class_aware_to_class_agnostic(class_aware_dir, folder_name = class_agn_folder)
    print('********* Class-Agnostic patches created successfully *************')
        


def make_random_patches_per_sheet(dataset_dir, folder_name='patches_class_aware', sz=1024, n=20):
    """
    Create random patches of specified size from images and corresponding YOLO annotations.

    Parameters:
    -----------
    dataset_dir : Path
        Dataset directory where full-size PID images and their annotations in YOLO format are present.
    dest_dir : str
        Destination directory where cropped image patches and updated annotations will be saved.
    sz : int, optional
        Size of the patch (height and width), default is 1024.
    n : int, optional
        Number of random patches per image, default is 20.

    Returns:
    --------
    None
    """
    
    im_pths, txt_pths = get_im_txt_pths(dataset_dir)
    dest_dir = Path(f'{dataset_dir.parent}/{folder_name}')
    os.makedirs(dest_dir, exist_ok=True)
    
    for im_pth, txt_pth in zip(im_pths, txt_pths):
        # Load image
        image = cv2.imread(str(im_pth))
        if image is None:
            print(f"Error reading image: {im_pth}")
            continue
        
        H, W = image.shape[:2]

        # Read YOLO annotations
        try:
            with open(txt_pth, 'r') as f:
                lines = [list(map(float, line.split())) for line in f.readlines()]
        except Exception as e:
            print(f"Error reading annotation file: {txt_pth} - {e}")
            continue

        arr = np.array(lines)
        class_labels = arr[:, 0].tolist()  # First column: class IDs
        bboxes = arr[:, 1:]               # Remaining columns: YOLO bounding boxes
        
        xmax = W - sz
        ymax = H - sz
        
        for i in range(n):
            # Randomly select a starting point for the patch
            x_start = random.randint(0, xmax)
            y_start = random.randint(0, ymax)

            # Define the transformation for cropping
            transform = A.Compose([
                A.Crop(x_min=x_start, y_min=y_start, x_max=x_start + sz, y_max=y_start + sz),
                A.Resize(sz, sz)
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5, label_fields=['class_labels']))

            # Apply the transformation
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

            # Prepare YOLO annotations for the transformed image
            yolo_bboxes = [f"{int(label)} {x_c} {y_c} {w} {h}" for (x_c, y_c, w, h), label in zip(transformed_bboxes, transformed_class_labels)]

            # Save the new annotations
            annotation_name = f"{im_pth.stem}_patch_{i}.txt"
            with open(os.path.join(dest_dir, annotation_name), 'w') as g:
                for bbox in yolo_bboxes:
                    g.write(f"{bbox}\n")

            # Save the cropped image patch
            patch_name = f"{im_pth.stem}_patch_{i}.jpg"
            cv2.imwrite(os.path.join(dest_dir, patch_name), transformed_image)
        
        #print(f"Processed {n} patches for image: {im_pth.name}")

    print('********* Random patches creation completed successfully *************')

def get_bboxes(im_pth, txt_pth):
    """
    Reads bounding box annotations from a YOLO format text file and 
    converts them to pixel coordinates for a given image.

    Parameters:
    -----------
    im_pth : str or Path
        Path to the image file.
    txt_pth : str or Path
        Path to the corresponding YOLO annotation file.

    Returns:
    --------
    bboxes : list of tuples
        A list of bounding boxes, where each box is represented as 
        (class_id, x_min, y_min, x_max, y_max).
    """
    
    # Load image and get its dimensions
    im = cv2.imread(str(im_pth))
    if im is None:
        raise ValueError(f"Image at {im_pth} could not be loaded.")
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    H, W = im.shape[:2]
    
    bboxes = []
    
    # Read YOLO annotation file
    with open(txt_pth, "r") as f:
        for line in f:
            class_id, x_c, y_c, w, h = map(float, line.split())
            # Convert YOLO format (center coordinates) to pixel coordinates
            x_c, y_c, w, h = (x_c * W, y_c * H, w * W, h * H)
            x_min = int(x_c - w / 2)
            y_min = int(y_c - h / 2)
            x_max = int(x_c + w / 2)
            y_max = int(y_c + h / 2)
            bboxes.append((int(class_id), x_min, y_min, x_max, y_max))
    
    return bboxes


def plot_bboxes(im_pth, txt_pth, num_classes=40):
    """
    Plots bounding boxes on an image with class-specific colors.

    Parameters:
    -----------
    im_pth : str or Path
        Path to the image file.
    txt_pth : str or Path
        Path to the corresponding YOLO annotation file.
    num_classes : int, optional
        Number of classes to generate unique colors for (default is 40).

    Returns:
    --------
    im : np.array
        Image with bounding boxes drawn on it.
    """
    
    # Load image
    im = cv2.imread(str(im_pth))
    if im is None:
        raise ValueError(f"Image at {im_pth} could not be loaded.")
    
    # Convert to RGB for consistent plotting
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Get bounding boxes and class labels
    bboxes = get_bboxes(im_pth, txt_pth)
    
    # Generate color map for class-specific bounding box colors
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / num_classes)[:3] for i in range(num_classes)]  # RGB values in range [0, 1]
    
    # Plot bounding boxes
    for (class_id, x_min, y_min, x_max, y_max) in bboxes:
        color = tuple([int(c * 255) for c in colors[int(class_id)]])  # Convert to BGR for OpenCV
        
        # Draw rectangle for bounding box
        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color, 2)
    
    return im


def plot_ims_labels_grid(dataset_dir, n=8, num_classes=100, figsize=(15,5)):
    """
    Plots a grid of images with bounding boxes overlaid.

    Parameters:
    -----------
    dataset_dir : Path
        Dataset directory where full-size PID images and their annotations in YOLO format are present.
    n : int, optional
        Number of images to display in the grid (default is 8).
    num_classes: int, optional
        Number of classes to generate unique colors for (default is 40).

    Returns:
    --------
    None
    """
    
    # Get image and annotation file paths
    im_pths, txt_pths = get_im_txt_pths(dataset_dir)
    
    # Randomly select `n` images from the dataset
    num_cols = min(len(im_pths), n)
    inds = random.sample(range(len(im_pths)), num_cols)  # Select random image indices
    
    rand_ims = [im_pths[i] for i in inds]
    rand_txts = [txt_pths[i] for i in inds]
    
    # Plot the images with bounding boxes
    ims = [plot_bboxes(im_pth, txt_pth, num_classes) for im_pth, txt_pth in zip(rand_ims, rand_txts)]
    
    # Display images in a grid
    plt.figure(figsize=figsize)
    show_images(ims, nrows=1, figsize=figsize)
    plt.show()
