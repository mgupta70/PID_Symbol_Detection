from fastai.vision.all import *
from natsort import natsorted
import cv2
import gc

### Stage-2 #############
import albumentations as A
from ultralytics import YOLO
import yaml
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
###########################



################
### stage-1 ####
################

def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    
def get_im_txt_pths(im_dir, txt_dir):
    ''' Inputs: im_dir and txt_dir containing yolo annotations '''
    im_pths = natsorted(get_files(im_dir, extensions='.jpg'))
    txt_pths = natsorted(get_files(txt_dir, extensions='.txt'))
    return im_pths, txt_pths
    
    
def copy_files_from_dirs(src_dir, dest_dir, ftype, to_run = False):
    ''' 
    Currently, it does not copy the folder structure from src_dir
    Inputs: 
    1. src_dir: source folder
    2. dest_dir: destination folder
    1. ftype: file type to copy
    ex: ftype = '.txt' or '.jpg' 
    '''
    
    os.makedirs(dest_dir, exist_ok=True)
    #files = glob.glob(f"{src_dir}/*{ftype}") # without fastai
    files = natsorted(get_files(src_dir, extensions=f'{ftype}')) # fastai
    print('found ',len(files) ,'files. Sample file: ', files[0])
    if to_run:
        print(f'------ moving {len(files)} {ftype} files ------')
        for file in files:
            name = Path(file).name
            dest_file = f"{dest_dir}/{name}"
            shutil.copy(file, dest_file)
        print(f'************ DONE  **************')
    else:
        print('COPY not enabled')

def copy_files_v2(file_pths, dest_dir, to_run = False):
    ''' copy files in file_pths list to  to dest_dir '''
    
    os.makedirs(dest_dir, exist_ok=True)
    if to_run:
        print(f'------ moving {len(file_pths)} files ------')
        for file in file_pths:
            name = Path(file).name
            dest_file = f"{dest_dir}/{name}"
            shutil.copy(file, dest_file)
        print(f'************ DONE  **************')
    else:
        print('COPY not enabled')
        
def xyxy2yolo(category, bbox):
    ''' category as int or str type; bbox as array of 4 numbers'''
    category = str(category)
    x_c = str((bbox[0]+bbox[2])/2)
    y_c = str((bbox[1]+bbox[3])/2)
    w = str(bbox[2]-bbox[0])
    h = str(bbox[3]-bbox[1])
    yolo_line = " ".join([category,x_c,y_c,w,h]) + '\n'
    return yolo_line


def dpid_npy2yolo(im_pths, to_run = False):
    ''' Input - image paths, Output -  yolo txt file | condition - .npy files be in same directory as im_pths '''
    if to_run:
        for fname in im_pths:
            print(f'processing: {fname.name}')
            im = cv2.imread(str(fname))
            H,W = im.shape[:2]
            symbol_file_pth = os.path.join(fname.parent, fname.name.replace('.jpg', '_symbols.npy'))
            data = np.load(symbol_file_pth, allow_pickle=True)
            data[:,2] = np.array(data[:,2], dtype = int) # labels are str and int type mix, let;s make all int
            normalise_denom = np.array([W,H,W,H])
            bbox_arr = np.array(list(data[:,1])) # xmin, ymin, xmax, ymax
            bboxes_normalised = (bbox_arr/normalise_denom).tolist() # xmin, ymin, xmax, ymax but normalised
            class_labels = (data[:,2]-1).tolist() # -1 to take care of class number
            
            txt_name = str(fname.name).replace('.jpg', '.txt')
            with open(f'{fname.parent}/{txt_name}', 'w+') as txt_file:  
                for category, bbox in zip(class_labels, bboxes_normalised):
                    line = xyxy2yolo(category, bbox)
                    txt_file.write(line)
                
                txt_file.close()
        print('*********************************')
    else:
        print('conversion to yolo txt is not running.')
        
        

def class_aware2class_agnostic(txt_pths, dest_dir, to_run=False):
    ''' txt_pths = class aware txt file paths| Only creates txt files and not copy images'''

    if to_run:
        os.makedirs(dest_dir, exist_ok=True)
        
        for file in txt_pths:
            with open(file,'r') as f:
                lines = f.readlines()
    
            with open(f"{dest_dir}/{file.name}", 'w+') as g:
                for line in lines:
                    bbox = line.split()[1:]
                    category = '0'
                    new_line = [category] + bbox # list
                    new_line = ' '.join(new_line) +'\n' # string
                    g.write(new_line)
            g.close()
            f.close()
        print('******** Done ********')
                
    else:
        print('Conversion is not running') 

        

def get_bboxes(im_pth, txt_pth):
    im = cv2.imread(str(im_pth))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    H,W = im.shape[:2]
    bboxes = []
    with open(txt_pth, "r") as f:
        for line in f:
            class_id, x_c, y_c, w, h = map(float, line.split())
            x_c, y_c, w, h = (x_c * W, y_c * H, w * W, h * H)
            x_min = int(x_c - w / 2)
            y_min = int(y_c - h / 2)
            x_max = int(x_c + w / 2)
            y_max = int(y_c + h / 2)
            bboxes.append((int(class_id), x_min, y_min, x_max, y_max))
    return bboxes


def plot_bboxes(im_pth,txt_pth):
    im = cv2.imread(str(im_pth))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    bboxes = get_bboxes(im_pth, txt_pth)
   
    for (class_id, x_min, y_min, x_max, y_max) in bboxes:
        red = np.linspace(255,0,45) # 40 classes for now
        green = np.linspace(0,255,45)
        blue = np.linspace(0,255,45)

        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (int(red[class_id]), int(green[class_id]), int(blue[class_id])), 2)

    return im

# def plot_bboxes(im_pth,txt_pth):
#     im = cv2.imread(str(im_pth))
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     H,W = im.shape[:2]
#     bboxes = []
#     with open(txt_pth, "r") as f:
#         for line in f:
#             class_id, x_c, y_c, w, h = map(float, line.split())
#             x_c, y_c, w, h = (x_c * W, y_c * H, w * W, h * H)
#             x_min = int(x_c - w / 2)
#             y_min = int(y_c - h / 2)
#             x_max = int(x_c + w / 2)
#             y_max = int(y_c + h / 2)
#             bboxes.append((int(class_id), x_min, y_min, x_max, y_max))

#     for (class_id, x_min, y_min, x_max, y_max) in bboxes:
#         red = np.linspace(255,0,40) # 40 classes for now
#         green = np.linspace(0,255,40)
#         blue = np.linspace(0,255,40)

#         cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (int(red[class_id]), int(green[class_id]), int(blue[class_id])), 2)

#     return im
def crop_image(img, coordinates):
    cropped_img = img[int(coordinates[1]):int(coordinates[3]), int(coordinates[0]):int(coordinates[2])]
    return cropped_img


def make_patches_w_overlap(im_pths, txt_pths, dest_dir, overlap=0.25, sz=1024, to_run = False):
    ''' overlap = 0.25 means 25%'''
    if to_run:
        os.makedirs(dest_dir, exist_ok=True)
        
        for im_pth, txt_pth in zip(im_pths, txt_pths):
            im = cv2.imread(str(im_pth))
            H,W = im.shape[:2]
        
            with open(txt_pth,'r') as f:
                lines = f.readlines()
                f.close()
            ls = [list(map(float,l.split())) for l in lines] # each line as a list of floats
            arr = np.stack(ls) # convert to array for easy indexing
            class_labels = arr[:,0].tolist()
            bboxes = arr[:,1:]
            
            for i in range(int(W//(sz*(1-overlap)))):
                for j in range(int(H//(sz*(1-overlap)))):
                    transform = A.Compose([
                        A.Crop(x_min=sz*i*overlap, y_min=sz*j*overlap, x_max=min(W,sz*(i*overlap+1)), y_max=min(H, sz*(j*overlap+1)))
                    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5, label_fields=['class_labels']))
        
                    transformed = transform(image=im, bboxes=bboxes, class_labels=class_labels)
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']
                    transformed_class_labels = transformed['class_labels']
                    yolo_bboxes = []
                    for box, label in zip(transformed_bboxes, transformed_class_labels):
                        x_c, y_c, w, h = box
                        yolo_bboxes.append(f"{int(label)} {x_c} {y_c} {w} {h}")
        
                    with open(f"{dest_dir}/{i}_{j}_{txt_pth.name}", 'w+') as g:
                        for bbox in yolo_bboxes:
                            g.write(f"{bbox}\n")
                        g.close()
                        
                    cv2.imwrite(f'{dest_dir}/{i}_{j}_{im_pth.name}', transformed_image)
        print('********* Done *************')
    else:
        print('Make patches is not running')
            
        
def make_classes_folders(src_dir, dest_dir, to_run = False):
    if to_run:
        os.makedirs(dest_dir, exist_ok=True)
        im_pths =  natsorted(get_files(src_dir, extensions='.jpg'))
        print('Total Images: ', len(im_pths))

        counter = 0
        im_counter = 0
        for im_pth in im_pths:
            im = cv2.imread(str(im_pth))
            H,W = im.shape[:2]
            im_name = im_pth.name[:-4]
            txt_name = im_name+'.txt'
            txt_pth = os.path.join(im_pth.parent, txt_name)
            with open(txt_pth, 'r') as f:
                lines = f.readlines()
                f.close()

            ls = [list(map(float,l.split())) for l in lines] # each line as a list of floats
            arr = np.stack(ls) # convert to array for easy indexing
            class_labels = arr[:,0].tolist()
            bboxes = arr[:,1:]
            xyxy_boxes = yolo2xyxy(bboxes,H,W)
            for class_id, box in zip(class_labels, xyxy_boxes):
                class_id = int(class_id)
                os.makedirs(f"{dest_dir}/{class_id}", exist_ok=True)
                cropped_im = crop_image(im, box)
                cv2.imwrite(f'{dest_dir}/{class_id}/{im_name}_{counter}.jpg',cropped_im)
                counter+=1

            im_counter+=1
            if im_counter%5==0:
                print(f'Processed {im_counter} drawings and saved {counter} symbols')
                
        print('************** DONE *****************')
    else:
        print('Making class wise folder not running')

def make_patches_per_sheet(im_pths, txt_pths, dest_dir, sz=1024, n = 20,  to_run = False):
    ''' make n patches per sheet of size sz'''
    if to_run:
        os.makedirs(dest_dir, exist_ok=True)
        
        for im_pth, txt_pth in zip(im_pths, txt_pths):
            im = cv2.imread(str(im_pth))
            H,W = im.shape[:2]
        
            with open(txt_pth,'r') as f:
                lines = f.readlines()
                f.close()
            ls = [list(map(float,l.split())) for l in lines] # each line as a list of floats
            arr = np.stack(ls) # convert to array for easy indexing
            class_labels = arr[:,0].tolist()
            bboxes = arr[:,1:]
            xmax = W - sz - 1
            ymax = H - sz - 1 
            
            for i in range(n):
                x_start = random.randint(0, xmax)
                y_start = random.randint(0, ymax)

                transform = A.Compose([
                    A.Crop(x_min=x_start, y_min=y_start, x_max=min(W,x_start+sz), y_max=min(H, y_start+sz))
                ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5, label_fields=['class_labels']))

                transformed = transform(image=im, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']

                yolo_bboxes = []
                for box, label in zip(transformed_bboxes, transformed_class_labels):
                    x_c, y_c, w, h = box
                    yolo_bboxes.append(f"{int(label)} {x_c} {y_c} {w} {h}")

                with open(f"{dest_dir}/{i}_{txt_pth.name}", 'w+') as g:
                    for bbox in yolo_bboxes:
                        g.write(f"{bbox}\n")
                    g.close()

                cv2.imwrite(f'{dest_dir}/{i}_{im_pth.name}', transformed_image)
        print('********* Done *************')
    else:
        print('Make random patches is not running')

    
    
def get_train_val_test_txts_full_OLD(src_dir, num_classes, num_train, num_val, to_run = False):
    ''' input: num_classes and num_train '''
    if to_run:
        im_pths =  natsorted(get_files(src_dir, extensions='.jpg'))
        txt_pths =  natsorted(get_files(src_dir, extensions='.txt'))

        total_ims = len(im_pths)
        total_txts = len(txt_pths)

        if total_ims!=total_txts:
            print('Number of images and txts dont match')
        
        num_test = total_ims - (num_train+num_val)
        print('Train:Val:Test', num_train, num_val, num_test)
        print(f"sampling {num_train} train images from a total of {total_ims} images")
        iteration = 0
        while True:
            train_txts = random.sample(txt_pths, k=num_train)
            classes_sheets = []
            for txt_pth in train_txts:
                with open(txt_pth, 'r') as f:
                    lines = f.readlines()
                ls = [list(map(float, l.split())) for l in lines]  # Each line as a list of floats
                if len(ls) > 0:
                    arr = np.stack(ls)  # Convert to array for easy indexing
                    class_labels = arr[:, 0]
                    unique_classes = np.unique(class_labels)
                    classes_sheets.append(unique_classes)

            # Concatenate all unique classes found
            all_unique_classes = np.unique(np.concatenate(classes_sheets))
            num_classes_sampled = all_unique_classes.shape[0]

            if num_classes_sampled == num_classes:
                print(f"Iteration {iteration}: {num_classes_sampled} unique classes sampled.")
                break

            iteration += 1
            if iteration > 1000:  # Prevent infinite loop
                print("Unable to sample within the given constraints after 1000 iterations.")
                break

        remaining_txts = list(set(txt_pths).difference(train_txts))
        val_txts = random.sample(remaining_txts, k=num_val)
        test_txts = list(set(remaining_txts).difference(val_txts))
        
        return train_txts, val_txts, test_txts
    

def get_k_files(txt_pths, k):
    
    k_txts = random.sample(txt_pths, k=k)
    classes_sheets = []
    for txt_pth in k_txts:
        with open(txt_pth, 'r') as f:
            lines = f.readlines()
        ls = [list(map(float, l.split())) for l in lines]  # Each line as a list of floats
        if len(ls) > 0:
            arr = np.stack(ls)  # Convert to array for easy indexing
            class_labels = arr[:, 0]
            unique_classes = np.unique(class_labels)
            classes_sheets.append(unique_classes)

    # Concatenate all unique classes found
    all_unique_classes = np.unique(np.concatenate(classes_sheets))
    all_unique_classes = [int(o) for o in all_unique_classes]
    return k_txts, list(all_unique_classes)
    
    
def get_train_val_test_txts_full(src_dir, num_train, num_val, main_classes, to_run = False):
    ''' inpu: num_classes and num_train '''
    if to_run:
        im_pths, txt_pths = get_im_txt_pths(src_dir, src_dir)

        total_ims = len(im_pths)
        total_txts = len(txt_pths)
        if total_ims!=total_txts:
            print('Number of images and txts dont match')
        
        num_test = total_ims - (num_train+num_val)
        print('Train:Val:Test', num_train, num_val, num_test)
        print(f"sampling {num_train} train images from a total of {total_ims} images")
        
        
        iteration = 0
        while True:
            train_txts, train_classes = get_k_files(txt_pths, num_train)
            remaining_txts = list(set(txt_pths).difference(train_txts))
            test_txts, test_classes = get_k_files(remaining_txts, num_test)
            val_txts = list(set(remaining_txts).difference(test_txts))
            
            train_satisfy = len(list(set(train_classes).intersection(set(main_classes))))==len(main_classes)
            test_satisfy = natsorted(list(set(test_classes).intersection(set(main_classes))))==natsorted(main_classes)

            print(train_satisfy, test_satisfy)
            if train_satisfy==True & test_satisfy==True:
                print(f"Iteration {iteration}: Done")
                break

            iteration += 1
            if iteration > 1000:  # Prevent infinite loop
                print("Unable to sample within the given constraints after 1000 iterations.")
                break

        return train_txts, val_txts, test_txts
    
def get_txt_patches_from_full(full_txts_pths, patches_dir):
    ''' for sundt'''
    
    full_txts_names = [o.name for o in full_txts_pths]
    all_patches_pths = natsorted(natsorted(get_files(patches_dir, extensions='.txt')))
    all_patches_names = [o.name for o in all_patches_pths]
    txts_to_move = []
    for f in full_txts_names:
        txts_to_move.append([Path(patches_dir  +'/' + o) for o in all_patches_names if f in o])
    flattened_list = []
    for ls in txts_to_move:
        for o in ls:
            flattened_list.append(o)
    return flattened_list

def make_yolo_training_data_patches(dest_dir, img_dir, txt_dir, train_txts, val_txts, test_txts, to_run= False):
    if to_run:
        os.makedirs(dest_dir, exist_ok=True)
        
        # move train files
        image_folder = dest_dir+'/images/train/';os.makedirs(image_folder, exist_ok=True)
        txt_folder = dest_dir+'/labels/train/';os.makedirs(txt_folder, exist_ok=True)
        
        for txt_pth in train_txts:
            
            f = txt_dir + '/' + txt_pth.name
            shutil.copy(f, txt_folder+txt_pth.name)
            im_name = txt_pth.name.replace('.txt','.jpg')
            im_pth = img_dir+'/' + im_name 
            shutil.copy(im_pth,image_folder+im_name )
            
        print('***** Moved Train files **********')
            
        # move val files
        image_folder = dest_dir+'/images/val/';os.makedirs(image_folder, exist_ok=True)
        txt_folder = dest_dir+'/labels/val/';os.makedirs(txt_folder, exist_ok=True)
        
        for txt_pth in val_txts:
            f = txt_dir + '/' + txt_pth.name
            shutil.copy(f, txt_folder+txt_pth.name)
            im_name = txt_pth.name.replace('.txt','.jpg')
            im_pth = img_dir+'/' + im_name 
            shutil.copy(im_pth,image_folder+im_name )
            
        print('***** Moved Val files **********')
        
        # move test files
        image_folder = dest_dir+'/images/test/';os.makedirs(image_folder, exist_ok=True)
        txt_folder = dest_dir+'/labels/test/';os.makedirs(txt_folder, exist_ok=True)
        
        for txt_pth in test_txts:
            f = txt_dir + '/' + txt_pth.name
            shutil.copy(f, txt_folder+txt_pth.name)
            im_name = txt_pth.name.replace('.txt','.jpg')
            im_pth = img_dir+'/' +im_name 
            shutil.copy(im_pth,image_folder+im_name )
            
        print('***** Moved Test files **********')
    else:
        print('Yolo folder making not running')
        
        
        


################
### stage-2 ####
################


def plot_class_distribution(classes_dir):
    ''' replace with yolo train+val txts''' 
    target_names = natsorted(os.listdir(classes_dir))
    target_counts = []
    for t in target_names:
        pth = os.path.join(classes_dir, t)
        num_images = len(get_image_files(pth))
        target_counts.append(num_images)


    fig = plt.figure(figsize = (10, 4))

    # creating the bar plot
    bars = plt.bar(target_names, target_counts, color ='maroon', width = 0.4)
    plt.xticks(rotation = 90)
    plt.xlabel("Symbol names")
    plt.ylabel("Count")
    plt.title("Frequency distribution")
    
    
    # Adding the counts above the bars
    for bar, count in zip(bars, target_counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), ha='center', va='bottom')

    plt.tight_layout()  # Adjust layout to make room for the labels    
    plt.show()
    
    
def perform_SAHI(src_dir, dest_dir, weights_file, slice_size, suffix = 'agnostic', with_conf_score = True, to_run = False):
    if to_run:
        im_pths = get_image_files(src_dir); print('Total test images: ', len(im_pths))
        
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
    

def yolo2xyxy(bboxes,H,W):
    ''' bboxes is an array of shape (N,4) for x_c, y_c, w,h in yolo format'''
    xmin = (bboxes[:,0] - bboxes[:,2]/2)*W
    ymin = (bboxes[:,1] - bboxes[:,3]/2)*H
    xmax = (bboxes[:,0] + bboxes[:,2]/2)*W
    ymax = (bboxes[:,1] + bboxes[:,3]/2)*H

    return np.stack([xmin,ymin,xmax,ymax], axis=1)

def GT_yolo2xyxy(src_dir, dest_dir, to_run = False):
    if to_run:
        os.makedirs(dest_dir, exist_ok=True)
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
            class_labels = arr[:,0]
            bboxes = arr[:,1:]
            xyxy_boxes = yolo2xyxy(bboxes,H,W)

            new_txt_pth = dest_dir+'/'+name
            with open(f'{new_txt_pth}', 'w+') as txt_file: 
                for class_id, box in zip(class_labels, xyxy_boxes):
                    class_id = np.array([class_id], dtype=int)
                    box = np.array(box, dtype=int)

                    l = np.concatenate([class_id, box]).tolist()
                    yolo_line = ' '.join(map(str, l)) + '\n'
                    txt_file.write(yolo_line)  
                txt_file.close()
        print('******* DONE yolo2xyxy*******')        
    else:
        print('yolo to xyxy not running')

            
def preds_yolo2xyxy(src_dir, dest_dir, to_run = False):
    if to_run:
        os.makedirs(dest_dir, exist_ok=True)
        txt_pths = natsorted(get_files(src_dir, extensions='.txt'))
        num_txts = len(txt_pths); print(f'converting {num_txts} files to xyxy format')
        for txt_pth in txt_pths:
            name = txt_pth.name
            im_pth = src_dir+'/'+name.replace('.txt','.jpg')
            im = cv2.imread(im_pth)
            H,W = im.shape[:2]
            with open(txt_pth, 'r') as f:
                #print('preds parsing: ',name)
                lines = f.readlines()
                f.close()

            ls = [list(map(float,l.split())) for l in lines] # each line as a list of floats
            arr = np.stack(ls) # convert to array for easy indexing
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
        print('******* DONE yolo2xyxy*******')
    else:
        print('yolo to xyxy not running')
        
def get_class_distribution(txt_pths):
    arrs = []
    for txt_pth in txt_pths:
        with open(txt_pth, 'r') as f:
            lines = f.readlines()
        ls = [list(map(float, l.split())) for l in lines]  # Each line as a list of floats
        if len(ls) > 0:
            arr = np.stack(ls)  # Convert to array for easy indexing
            class_labels = list(arr[:, 0])
            arrs.append(class_labels)

    flatten_arrs = []
    for l in arrs:
        for x in l:
            flatten_arrs.append(x)
            
    return flatten_arrs

def plot_class_distributionv2(flatten_distribution):
    plt.figure(figsize=(12,6))
    c = Counter(flatten_distribution)
    bars = plt.bar(c.keys(), c.values(), color ='maroon', width = 0.6)
    plt.xticks(list(c.keys()), rotation = 0)
    plt.xlabel("Symbol names")
    plt.ylabel("Count")
    # Adding the counts above the bars
    for bar, count in zip(bars, c.values()):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), ha='center', va='bottom')

    plt.tight_layout()  # Adjust layout to make room for the labels 
    plt.show()
