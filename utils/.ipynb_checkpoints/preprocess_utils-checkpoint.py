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

def make_patches_w_overlap(dataset_dir, overlap=0.25, sz=1024):
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

    # im_pths =  natsorted(get_files(dataset_dir, extensions='.jpg'))
    # txt_pths =  natsorted(get_files(dataset_dir, extensions='.txt'))
    im_pths, txt_pths = get_im_txt_pths(dataset_dir)
    dest_dir = f'{dataset_dir.parent}/patches_class_aware'
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
                    A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5, label_fields=['class_labels']))

                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']

                # Save the cropped image
                cropped_img_name = f"{i}_{j}_{im_pth.stem}.jpg"
                cv2.imwrite(f"{dest_dir}/{cropped_img_name}", transformed_image)

                # Save the corresponding YOLO annotation
                annotation_name = f"{i}_{j}_{txt_pth.stem}.txt"
                with open(f"{dest_dir}/{annotation_name}", 'w') as g:
                    for box, label in zip(transformed_bboxes, transformed_class_labels):
                        x_c, y_c, w, h = box
                        g.write(f"{int(label)} {x_c} {y_c} {w} {h}\n")
        
        print(f"Processed patches for image: {im_pth.name}")
    
    print('********* All patches created successfully *************')


def make_random_patches_per_sheet(dataset_dir, sz=1024, n=20):
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
    
    # im_pths =  natsorted(get_files(dataset_dir, extensions='.jpg'))
    # txt_pths =  natsorted(get_files(dataset_dir, extensions='.txt'))
    im_pths, txt_pths = get_im_txt_pths(dataset_dir)
    dest_dir = f'{dataset_dir.parent}/patches_class_aware'
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
                A.Crop(x_min=x_start, y_min=y_start, x_max=x_start + sz, y_max=y_start + sz)
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
        
        print(f"Processed {n} patches for image: {im_pth.name}")

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


def plot_ims_labels_grid(dataset_dir, n=8, num_classes=40):
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
    plt.figure(figsize=(15, 5))
    show_images(ims, nrows=1)
    plt.show()
    
