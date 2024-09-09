from ultralytics import YOLO
import torch

def train_yolo_model(yaml_filename, epochs=50, patience=5, batch_size=8, imgsz=1024, model_weights='yolov8n.pt'):
    """
    Function to train a YOLOv8 model with specified parameters.
    
    Args:
        yaml_filename (str): Path to the YAML file for training configuration.
        epochs (int): Number of training epochs. Default is 50.
        patience (int): Number of epochs to wait before early stopping. Default is 5.
        batch_size (int): Size of the batch for training. Default is 8.
        imgsz (int): Image size for training. Default is 1024.
        model_weights (str): Path to the pretrained model weights file. Default is 'yolov8n.pt'.
    
    Returns:
        None
    """
    
    # Determine the device to use
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the pretrained YOLOv8n model
    model = YOLO(model_weights)  
    
    # Train the model with the specified parameters
    model.train(
        data=yaml_filename, 
        epochs=epochs, 
        patience=patience, 
        batch=batch_size, 
        imgsz=imgsz, 
        device=device
    )
    
    print(f"Training completed for model with {epochs} epochs on {device}.")


# Example usage:
if __name__ == "__main__":
    train_yolo_model(
        yaml_filename='asupid_class_agnostic_train.yaml', 
        epochs=50, 
        patience=5, 
        batch_size=8, 
        imgsz=1024, 
        model_weights='yolov8n.pt'
    )
