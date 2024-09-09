import os, sys
sys.path.append(os.path.abspath('..'))
from utils.evaluation_utils import *

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_result(gt_dir, pred_dir):
    
    # Convert Yolo annotations to XYXY format (required to calculate metrics)
    GT_xyxy_dir = convert_to_xyxy(src_dir=gt_dir, is_gt=True)
    preds_xyxy_dir = convert_to_xyxy(src_dir=pred_dir, is_gt=False)
    
    # Calculate Metrics
    get_detection_metrics(GT_xyxy_dir, preds_xyxy_dir, dataset_dir=preds_xyxy_dir.parent, df_name=f'Results_{preds_xyxy_dir.name}.csv')


if __name__ == "__main__":

    # Default parameters for Class-Aware Testing
    gt_dir = Path('../data/sample_dataset/test_set')
    pred_dir = Path('../data/sample_dataset/test_set_label_transfer')

    # Evaluate
    evaluate_result(gt_dir, pred_dir)
    