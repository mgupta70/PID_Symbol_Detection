import os, sys
sys.path.append(os.path.abspath('..'))
from utils.stage1_utils import *


def run_yolo_with_sahi(test_dir, yolo_weights, suffix='agnostic', slice_size=1024):
    
    im_pths, _ = get_im_txt_pths(dataset_dir=test_dir)

    # run SAHI
    perform_SAHI(im_pths=im_pths, 
                 weights_file=yolo_weights, 
                 sz = slice_size, 
                 suffix = suffix, 
                 with_conf_score = True, 
                 to_run = True)


    
if __name__ == "__main__":
    
    # Default parameters
    root_dir = Path('../data/sample_dataset/')
    test_dir =  Path(f"{root_dir}/test_set")

    # run SAHI inference with Yolo model trained with Class-Agnostic labels
    yolo_weights = '../models/sample_class_agnostic/best.pt'
    run_yolo_with_sahi(test_dir=test_dir, yolo_weights, suffix='agnostic', slice_size=1024)

    # run SAHI inference with Yolo model trained with Class-Aware labels
    yolo_weights = '../models/sample_class_aware/best.pt'
    run_yolo_with_sahi(test_dir=test_dir, yolo_weights, suffix='aware', slice_size=1024)