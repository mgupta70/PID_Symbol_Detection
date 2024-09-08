from utils.preprocess_utils import *
import shutil
from pathlib import Path
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
from fastai.vision.all import *


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

def train_yolo_model(yaml_filename, project='../models/trained_yolo_models', epochs=50, patience=5, batch_size=8, imgsz=1024, pretrained_weights='../models/yolov8n.pt'):
    """
    Function to train a YOLOv8 model with specified parameters.
    
    Args:
        yaml_filename (str): Path to the YAML file for training configuration.
        project (str) = Name of the project directory where training outputs are saved. 
        epochs (int): Number of training epochs. Default is 50.
        patience (int): Number of epochs to wait before early stopping. Default is 5.
        batch_size (int): Size of the batch for training. Default is 8.
        imgsz (int): Image size for training. Default is 1024.
        pretrained_weights (str): Path to the pretrained model weights file. Default is '../models/yolov8n.pt'.
    
    Returns:
        None
    """
    
    # Determine the device to use
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the pretrained YOLOv8n model
    model = YOLO(pretrained_weights)  
    
    # Train the model with the specified parameters
    model.train(
        data=yaml_filename, 
        epochs=epochs, 
        patience=patience, 
        batch=batch_size, 
        imgsz=imgsz, 
        project=project,
        device=device
    )
    
    print(f"Training completed for model with {epochs} epochs on {device}.")

def predict_YOLO(yaml_filename, trained_weights, iou=0.5, conf=0.25, batch=1, imgsz=1024):
    # load weights of trained model
    model = YOLO(trained_weights)
    # Customize validation settings
    validation_results = model.val(data='{}'.format(yaml_filename), imgsz=imgsz, batch=batch, conf=conf, iou=iou, device=device)

############
### SAHI ###
############

def perform_SAHI(im_pths, weights_file, slice_size, suffix = 'agnostic', with_conf_score = True, to_run = False):
    if len(im_pths)>0:
        dest_dir = Path(f"{im_pths[0].parent.parent}/SAHI_results_{suffix}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f'Results will be saved to: {dest_dir}')
    if to_run:
        #im_pths = get_image_files(src_dir)
        print('Total test images: ', len(im_pths))
        
        # Sliced inference
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=f'{weights_file}',
            confidence_threshold=0.5,
            device="cuda:0",  # or 'cuda:0'
        )

        for im_pth in im_pths:
            name = im_pth.name; print('SAHI processing: ', name)
            im = cv2.imread(str(im_pth))
            H,W = im.shape[:2]

            result = get_sliced_prediction(
                f"{str(im_pth)}",
                detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25)

            result.export_visuals(export_dir=f"{dest_dir}/", file_name= f"{name[:-4]}_{suffix}", hide_labels = True)

            txt_pth = str(dest_dir)+'/'+ name.replace(".jpg", '.txt')

            with open(f'{txt_pth}', 'w+') as txt_file:
                for res_dict in result.to_coco_annotations():
                    bbox_coco = res_dict['bbox'] # x-left, y-left, w,h
                    score = str(res_dict['score'])
                    category = str(res_dict['category_id'])
                    x_c = str((bbox_coco[0]+bbox_coco[2]/2)/W) # str((bbox_coco[0]+bbox_coco[2]/2)/7168)
                    y_c = str((bbox_coco[1]+bbox_coco[3]/2)/H) #str((bbox_coco[1]+bbox_coco[3]/2)/4561)
                    h = str(bbox_coco[3]/H)
                    w = str(bbox_coco[2]/W)
                    if with_conf_score:
                        line = " ".join([category, score, x_c,y_c,w,h])+'\n'
                    else:
                        line =  " ".join([category, x_c,y_c,w,h])+'\n'
                    txt_file.write(line)
            txt_file.close()
        print('********* DONE - SAHI *********')
    else:
        print('SAHI not running')




