# PID_Symbol_Detection

### Proposed Framework vs Conventional Framework

<img src="./media/workflow.svg" >

### Benefits of Proposed Framework
Class-Agnostic Object Detection & One-shot Label Transfer is found to be more:
1. Generalizable to different underlying P&ID drawing styles
2. Robust to class-imbalance
compared to equivalent class-aware counterparts.

### Simplified Visual Walkthrough of Proposed Framework 

#### 1. Data preprocessing

This step breaks down large P&ID sheets into overlapping patches. 

<img src="./media/overlapping_patches.png" width="800">

Plus, class-aware labels are transformed into class-agnostic to prepare for training a Yolo object detection model.

#### 2. Train Yolo (Stage-1)

Trains a 'Generic' symbol detector

<img src="./media/train_yolo.svg" width="400">

#### 3. Inferencing with SAHI (Stage-1)

For large P&IDs infer on smaller patches and combine the results (implemented via <a href="https://github.com/obss/sahi"> SAHI </a>).

<img src="./media/sahi_sample.gif" width="250">

#### 4. Label Transfer (Stage-2)
Train a model using one labeled image per symbol class (e.g. P&ID legend). The model can be a Siamese Network/ Prototypical (Zero-shot) Network or a Traditional classifier trained on augmented images.

<img src="./media/label_transfer.png" width="400">

If you use this package in your work, please cite it as:
```
@article{GUPTA2024105260,
title = {Semi-supervised symbol detection for piping and instrumentation drawings},
journal = {Automation in Construction},
volume = {159},
pages = {105260},
year = {2024},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2023.105260},
url = {https://www.sciencedirect.com/science/article/pii/S0926580523005204},
author = {Mohit Gupta and Chialing Wei and Thomas Czerniawski},
}
```
