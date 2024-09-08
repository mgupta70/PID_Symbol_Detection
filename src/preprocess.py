import os, sys
sys.path.append(os.path.abspath('..'))
from utils.preprocess_utils import *

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def preprocess_data(root_dir: Path, original_dir: str, overlap: float = 0.1, sz: int = 1024, n_random_patches: int = 0):
    """
    Prepares the dataset for training by converting class-aware annotations to class-agnostic and generating overlapping patches.

    Parameters:
    - root_dir (Path): Root directory containing the dataset.
    - original_dir (str): Subdirectory within root_dir containing the original images and YOLO annotations.
    - overlap (float): Overlap percentage for patch generation. Default is 0.1 meaning 10%.
    - sz (int): Size of each patch. Default is 1024.
    - n_random_patches (int): Number of random patches to create. Default is 0.

    Returns:
    None
    """
    # Path to the original dataset containing full PID images and YOLO annotation files
    dataset_dir =  Path(f"{root_dir}/{original_dir}")

    # Step 1: Convert class-aware dataset to class-agnostic dataset
    class_aware_to_class_agnostic(dataset_dir, folder_name='original_class_agnostic')

    # Step 2: Generate overlapping patches from the full-sized PIDs (both class-agnostic & class-aware)
    make_patches_w_overlap(
        dataset_dir,
        overlap=overlap,
        sz=sz,
        n_random_patches=n_random_patches
    )

if __name__ == "__main__":
    # Default parameters
    root_dir = Path('../data/sample_dataset/')
    original_dir = 'original'

    # Prepare the dataset
    preprocess_data(root_dir, original_dir)
    