# PID_Symbol_Detection

### Proposed Framework vs Conventional Framework
<img src="./media/overview2.svg" >

### Benefits of Proposed Framework
Class-Agnostic Object Detection & One-shot Label Transfer is found to be more:
1. Generalizable to different underlying P&ID drawing styles
2. Robust to class-imbalance
compared to equivalent class-aware counterparts.

### Simplified Visual Walkthrough of Proposed Framework 

#### 1. Data preprocessing
<img src="./media/overlapping_patches.png" width="800">


#### 2. Train Yolo (Stage-1)
<img src="./media/train_yolo.svg" width="400">

#### 3. Inferencing with SAHI (Stage-1)
<img src="./media/sahi_sample.gif" width="250">

#### 4. Label Transfer (Stage-2)
