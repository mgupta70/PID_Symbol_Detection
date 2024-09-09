from fastai.vision.all import *
from IPython.display import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

#############
from utils.BoundingBox import BoundingBox
from utils.BoundingBoxes import BoundingBoxes
from utils.Evaluator import *
from utils.utils import *
##############
device = 'cuda' if torch.cuda.is_available() else 'cpu'




def getBoundingBoxes(GT_dir, pred_dir):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    
    files = natsorted(get_files(GT_dir, extensions='.txt'))
    allBoundingBoxes = BoundingBoxes()
    
    for f in files:
        nameOfImage = f.name.replace(".txt", "") # this should be taken care of!!
        f = str(f)
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])
            y = float(splitLine[2])
            x2 = float(splitLine[3])
            y2 = float(splitLine[4])
            bb = BoundingBox(nameOfImage,idClass, x, y, x2,y2, CoordinatesType.Absolute, (7168, 4561),BBType.GroundTruth,
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
        
    # Read detections
    files = natsorted(get_files(pred_dir, extensions='.txt'))
    for f in files:
        
        nameOfImage = f.name.replace(".txt", "")
        f = str(f)
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            #print(splitLine)
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            x2 = float(splitLine[4])
            y2 = float(splitLine[5])
            bb = BoundingBox(nameOfImage,idClass,x,y,x2,y2,CoordinatesType.Absolute, (7168,4561),BBType.Detected,confidence,
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes


def get_detection_metrics(GT_bboxes_xyxy, preds_bboxes_xyxy, dataset_dir, df_name):
    ''' res_df_name = 'sampleds_awr' '''
    
    ##### Metrics ########

    boundingboxes = getBoundingBoxes(GT_bboxes_xyxy, preds_bboxes_xyxy)
    evaluator = Evaluator()

    # metrics
    metricsPerClass = evaluator.GetPascalVOCMetrics(
        boundingboxes, IOUThreshold=0.5, method=MethodAveragePrecision.EveryPointInterpolation) 

    columns=['class', 'total_positives', 'TP', 'FP', 'recall', 'precision']

    df = pd.DataFrame(columns = columns)

    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    for mc in metricsPerClass:
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        total_p = mc['total positives']
        TP = mc['total TP']
        FP = mc['total FP']
        # Print AP per class
        if recall.shape[0]==0:
            print(f'OOPS: class {c}') # no single entry
            row = pd.DataFrame([[c, total_p, TP, FP, '-123', '-123']], columns=columns)
        else:
            print(f'Results: class {c}, Recall {recall[-1]}, Precision: {precision[-1]}, Average_precision {average_precision}')
            row = pd.DataFrame([[c, total_p, TP, FP, recall[-1], precision[-1]]], columns=columns)

        df = pd.concat([df, row], ignore_index=True)

    df.to_csv(f'{dataset_dir}/{df_name}', index=False)
    return df

def transfer_labels(src_dir, dest_dir, weights_file, num_classes):
    """src_dir: preds_agn, 
    dest_dir: preds_agn2awr"""

    os.makedirs(dest_dir, exist_ok=True)

    to_tensor = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # helps in removing the affect of color
    ])

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_file))
    model.eval()

    txt_pths = natsorted(get_files(src_dir, extensions='.txt'))
    for txt_pth in txt_pths:
        name = txt_pth.name
        im_pth = str(src_dir)+'/'+name.replace('.txt', '.jpg')
        im = cv2.imread(str(im_pth))
        H,W = im.shape[:2]
        with open(txt_pth, 'r') as f:
            lines = f.readlines()
            f.close()
        ls = [list(map(float,l.split())) for l in lines] # each line as a list of floats


        new_txt_pth = dest_dir+'/'+name
        with open(f'{new_txt_pth}', 'w+') as txt_file:  
            for l in ls:
                class_id, score, xc,yc,w,h = l
                xmin = (xc-w/2)*W
                xmax = (xc+w/2)*W
                ymin = (yc-h/2)*H
                ymax = (yc+h/2)*H
                cropped_im =crop_image(im, [xmin,ymin,xmax,ymax])

                image_pil = Image.fromarray(cropped_im.astype('uint8'))  # Convert to PIL format
                image_transformed = to_tensor(image_pil)
                image_transformed = image_transformed.unsqueeze(0)  # Add a batch dimension at index 0

                with torch.no_grad():
                    predictions = model(image_transformed)
                    _, new_class = torch.max(predictions, 1) # tensor[20]
                    new_class = new_class.item() # 20

                    # update
                    l[0] = new_class
                    yolo_line = ' '.join(map(str, l)) + '\n'
                    txt_file.write(yolo_line)           
        txt_file.close()
    print('*** Transferring Labels done ***')

def yolo2xyxy(bboxes,H,W):
    ''' bboxes is an array of shape (N,4) for x_c, y_c, w,h in yolo format'''
    xmin = (bboxes[:,0] - bboxes[:,2]/2)*W
    ymin = (bboxes[:,1] - bboxes[:,3]/2)*H
    xmax = (bboxes[:,0] + bboxes[:,2]/2)*W
    ymax = (bboxes[:,1] + bboxes[:,3]/2)*H

    return np.stack([xmin,ymin,xmax,ymax], axis=1)

def convert_to_xyxy(src_dir, is_gt=True):
    'is_gt = if ground truth , if False it means it is predictions which also has confidence values in each line of annotation'

    # make folder to save annotations
    dest_dir = Path(f"{src_dir.parent}/{src_dir.name}_xyxy")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    
    txt_pths = natsorted(get_files(src_dir, extensions='.txt'))
    num_txts = len(txt_pths); print(f'converting {num_txts} files to xyxy format')
    for txt_pth in txt_pths:
        name = txt_pth.name
        im_pth = src_dir+'/'+name.replace('.txt','.jpg')
        im = cv2.imread(im_pth)
        H,W = im.shape[:2]
        with open(txt_pth, 'r') as f:
            #print('GT parsing: ',name)
            lines = f.readlines()
            f.close()

        ls = [list(map(float,l.split())) for l in lines] # each line as a list of floats
        arr = np.stack(ls) # convert to array for easy indexing
        
        if is_gt:
            class_labels = arr[:,0]
            bboxes = arr[:,1:]
            xyxy_boxes = yolo2xyxy(bboxes,H,W)
    
            new_txt_pth = str(dest_dir)+'/'+name
            with open(f'{new_txt_pth}', 'w+') as txt_file: 
                for class_id, box in zip(class_labels, xyxy_boxes):
                    class_id = np.array([class_id], dtype=int)
                    box = np.array(box, dtype=int)
                    l = np.concatenate([class_id, box]).tolist()
                    yolo_line = ' '.join(map(str, l)) + '\n'
                    txt_file.write(yolo_line)  
                txt_file.close()
        else:
            class_labels = arr[:,0]
            scores = arr[:,1];#ones_array = np.ones_like(scores)
            bboxes = arr[:,2:]
            xyxy_boxes = yolo2xyxy(bboxes,H,W)
    
            new_txt_pth = dest_dir+'/'+name
            with open(f'{new_txt_pth}', 'w+') as txt_file: 
                for class_id, conf_score, box in zip(class_labels, scores, xyxy_boxes):
                    class_id = int(class_id)  # Ensure class_id is an integer
                    conf_score = float(conf_score)  # Ensure confidence score is a float
                    box = box.astype(int)  # Ensure bounding box coordinates are integers
                    l = [class_id, conf_score] + box.tolist()
                    yolo_line = ' '.join(map(str, l)) + '\n'
                    txt_file.write(yolo_line)
                txt_file.close()
            
    print('Annotations converted from Yolo to XYXY format and saved in : ', dest_dir)
    print('******* DONE yolo2xyxy *******')       
    return dest_dir


            
