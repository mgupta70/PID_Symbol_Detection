from utils.preprocess_utils import *
import shutil
from pathlib import Path

def make_train_val_test_split(dataset_dir, train_val_test_ratio=[0.64, 0.16, 0.20]):
    """
    Splits the dataset into training, validation, and test sets based on the specified ratio.

    Parameters:
    -----------
    dataset_dir : str or Path
        Directory containing the full dataset (images and corresponding YOLO annotations).
    train_val_test_ratio : list, optional
        Ratio for splitting the dataset into train, validation, and test sets. 
        Default is [0.64, 0.16, 0.20].

    Returns:
    --------
    tuple
        A tuple containing lists of image paths for training, validation, and test sets.
    """
    # Retrieve image paths from the dataset directory
    im_pths, _ = get_im_txt_pths(dataset_dir)
    
    total_images = len(im_pths)
    num_train = round(total_images * train_val_test_ratio[0])
    num_val = round(total_images * train_val_test_ratio[1])
    num_test = total_images - (num_train + num_val)
    
    print(f"Total images: {total_images}")
    print(f"Train: {num_train}, Val: {num_val}, Test: {num_test}")
    
    # Randomly sample images for each split
    train_images = random.sample(im_pths, k=num_train)
    remaining_images = list(set(im_pths) - set(train_images))
    val_images = random.sample(remaining_images, k=num_val)
    test_images = list(set(remaining_images) - set(val_images))
    
    return train_images, val_images, test_images

def select_crops(full_pid_splits, crops_folder_pth):
    """
    Selects and groups matching cropped images for each full PID image.

    Parameters:
    -----------
    full_pid_splits : list of lists
        A list containing lists of full PID image paths.
    crops_folder_pth : str or Path
        Path to the folder containing the cropped images.

    Returns:
    --------
    crops_splits : list of lists
        A list containing lists of cropped image paths, corresponding to each set of full PID images.
    """
    crop_im_pths, _ = get_im_txt_pths(crops_folder_pth)
    crops_splits = []
    
    for im_pths in full_pid_splits:
        matching_crops = []
        
        for im_pth in im_pths:
            # Find crops that match the current full image name
            name = im_pth.stem
            crops_for_this_image = [crop_pth for crop_pth in crop_im_pths if crop_pth.stem.startswith(name)]
            matching_crops.extend(crops_for_this_image)
        
        crops_splits.append(matching_crops)
    
    return crops_splits

def make_yolo_folders(dir_name, train_images, val_images, test_images):
    """
    Creates YOLO format folder structure and populates it with the split datasets.

    Parameters:
    -----------
    dir_name : str or Path
        Name of the directory to create for the YOLO dataset structure.
    train_images : list
        List of image paths for the training set.
    val_images : list
        List of image paths for the validation set.
    test_images : list
        List of image paths for the test set.
    """
    base_dir = Path(train_images[0]).parent.parent / dir_name
    base_dir.mkdir(parents=True, exist_ok=True)

    yolo_folders = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for folder, image_paths in yolo_folders.items():
        image_folder = base_dir / f'images/{folder}'
        label_folder = base_dir / f'labels/{folder}'
        image_folder.mkdir(parents=True, exist_ok=True)
        label_folder.mkdir(parents=True, exist_ok=True)
        
        for image_path in image_paths:
            shutil.copy(image_path, image_folder / image_path.name)
            label_path = image_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy(label_path, label_folder / label_path.name)
        
        print(f'Copied {len(image_paths)} files to {folder} folder')


