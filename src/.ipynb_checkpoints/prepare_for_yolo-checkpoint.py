import os, sys
sys.path.append(os.path.abspath('..'))
#from utils.helpers import *
from utils.stage1_utils import *

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_yolov8_folder_structure(root_dir, original_dir, has_patches = True, train_val_test_ratio=[0.64, 0.16, 0.2]):
    
    # Path to the original dataset containing full PID images and YOLO annotation files
    dataset_dir =  Path(f"{root_dir}/{original_dir}")

    # Make train, val, test sets from full PIDs¶
    train_ims, val_ims, test_ims = make_train_val_test_split(dataset_dir, train_val_test_ratio=train_val_test_ratio)
    
    # Move test ims and annotations into folder named "test_set"
    test_txts = [f"{o.parent}/{o.stem}.txt" for o in test_ims] # Annotation files
    test_dir =  Path(f"{root_dir}/test_set")
    test_dir.mkdir(parents=True, exist_ok=True)
    copy_files_to_directory(test_ims, test_dir) # copy test images
    copy_files_to_directory(test_txts, test_dir) # copy test annotations
    print(f"Successfully moved {len(test_ims)} test sheets & their annotations to {test_dir}")

    # Make folders for Yolo training
    if has_patches:
        # Case-1: Make train & val folders by selecting crops from full PIDs - for class_aware
        crops_folder_pth = Path(f'{root_dir}/patches_class_aware/')
        train_crops, val_crops = select_crops(train_val_ims = [train_ims, val_ims], crops_folder_pth=crops_folder_pth)
        # make yolo folders
        make_yolo_folders(dir_name='yolo_class_aware', train_images=train_crops, val_images=val_crops)
        
        # Case-2: Make train & val folders by selecting crops from full PIDs - for class_agnostic
        crops_folder_pth =  Path(f'{root_dir}/patches_class_agnostic/')
        train_crops, val_crops = select_crops(train_val_ims = [train_ims, val_ims], crops_folder_pth=crops_folder_pth)
        # make yolo folders
        make_yolo_folders(dir_name='yolo_class_agnostic', train_images=train_crops, val_images=val_crops)
    else:
        # for training Yolo on full pids (Not recommended) because A) Dataset becomes small and B) Images are of large size)
        make_yolo_folders(dir_name='yolo_full', train_images=train_ims, val_images=val_ims)
    print('*** FOLDER STRUCTURE IS READY TO TRAIN A YOLOV8 MODEL ***')
    
    
if __name__ == "__main__":
    # Default parameters
    root_dir = Path('../data/sample_dataset/')
    original_dir = 'original'

    # Prepare the dataset
    prepare_yolov8_folder_structure(root_dir, original_dir)
    