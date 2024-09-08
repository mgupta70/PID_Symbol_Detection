from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict

def train_yolo_model(yaml_filename, project='../models/trained_yolo_models' epochs=50, patience=5, batch_size=8, imgsz=1024, pretrained_weights='../models/yolov8n.pt'):
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

