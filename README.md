# 12507345_Multi-class-segmentation-of-tree-species
The project mainly focuses on distinguishing multiple tree species from UAV orthophotos using a convolutional neural network (CNN) architecture, specifically the U-Net model.

## Exercise 2 – Hacking

### What I implemented
I downloaded high‑resolution orthomosaic data from Zenodo(https://zenodo.org/records/15168163) and manually created tree crown polygons for multiple tree species using QGIS (https://qgis.org/project/overview/). This required learning QGIS , resolving shapefile issues, and carefully delineating individual tree crowns per species.  I used the polygons available from the data itself for some spruce trees and drew others myself.

Total number of tree crowns dilineated:-
-Beech :81 polygons
-Pine :138 polygons
-Birch :2 polygons
-Spruce :202 polygons

Based on these , I implemented a complete data set pipeline that (see `dataset_pipeline` folder):-
- Tiles the orthomosaic into fixed-size image patches,    
- rasterises species-specific crown polygons into semantic segmentation masks, and  
- automatically splits the data into training (70%), validation(15%), and test sets(15%).  

Based on this dataset, I then implemented a separate model training pipeline in PyTorch (see `model_training_pipeline` folder). This pipeline includes a small U-Net–based semantic segmentation model, data loading , a full training and validation loop, evaluation on a test set, computation of IoU and Dice metrics, and visualization of predictions.

### Exploration and experiments
The initial setup included six classes, including locusts. However, the mean intersection over union (IoU) fell below the target threshold of 0.40, despite experimenting with data augmentation using Albumentations, firstly with 20 epochs and then with 30 epochs. 

To improve robustness and performance, the setup was simplified to five classes (Background (class 0)  Beech (class 1)  Pine (class 2)  Birch (class 3)  Spruce (class 4)) and the dataset was regenerated accordingly. This reduced class confusion and led to improved and more stable Mean IoU and Dice scores, which are reported in the final results (`results/metrics.txt`).the final model was trained for 30 epochs, selected based on the training and validation loss curves  

Summary of experiments:  
- With 6 classes and no augmentation, the Mean IoU on the test set was approximately 0.20, clearly below the 0.40 target.
- 6 classes, with Albumentations: No significant improvement.  
- After adding data augmentation with Albumentations, the Mean IoU remained at a similar level and did not significantly improve.  
- After reducing the task to 5 classes and regenerating the dataset, clear improvement in Mean IoU and Dice scores (final result reported in `results/metrics.txt`). Here also I tried first with 20 epochs and then 30 epoch according to the training and validation loss curve.


### Metrics and evaluation
- **Error metrics:** Mean IoU per class and Mean Dice coefficient per class.  
- **Evaluation protocol:** Metrics computed on a held‑out test set that was never used for training or validation.  
- **Target:** Mean IoU ≥ 0.40.  

**Achieved results (test set):**  
- Mean IoU per class:  
  `[0.9197, 0.1472, 0.7517, 0.0019, 0.5711]`  
- Mean Dice per class:  
  `[0.9580, 0.2443, 0.8569, 0.0038, 0.7125]`
  
Although not all classes reach the target IoU individually, the model performs strongly for several species.Per‑image and per‑class metrics are reported in `results/metrics.txt`.  

### Qualitative results
Qualitative predictions were generated for selected test tiles. For each example, a single combined figure visualises the input RGB image, the ground‑truth semantic mask and the predicted semantic mask. All qualitative outputs are stored in the `results/` folder.
Each prediction uses a consistent colour legend that maps mask colours to tree species. Visual inspection reveals that the model learns coherent crown shapes and spatial boundaries.  

### Training behavior
The training and validation loss curves (`results/loss_curve.png`) show stable convergence over 30 epochs without severe overfitting. Training was performed on CPU, resulting in longer training times but consistent optimisation behaviour,the total training time for the final model was approximately 316 minutes.

### Deliverables
The repository contains the following artifacts:

- **Trained model weights:**  
  - `results/unet_model_final.pth`  

- **Training diagnostics:**  
  - training vs. validation loss curve: `results/loss_curve.png`  

- **Quantitative evaluation:**  
  - per‑class IoU and Dice scores on the test set: `results/metrics.txt`  

- **Qualitative predictions:**  
  - side‑by‑side visualizations of image, ground truth and prediction: `results/*.png`  

### Reproducibility and data access
To ensure reproducibility, the following are provided:  
- Jupyter notebooks and Python scripts for dataset generation, training, and evaluation.  
- In addition, a Google Drive folder is shared containing the generated dataset and the notebooks used during development and experimentation. (https://drive.google.com/drive/folders/1mjh-fzXv-imMOXHhI8UL34lrEdEy3jiV?usp=sharing) 

### Time spent
- Data preparation (data download, QGIS labeling, fixing shapefile issues): 15-16 hours  
- Dataset pipeline (tiling, mask generation, splits, debugging): 8 hours  
- Model and training pipeline (implementation, experiments, tuning, reducing to five classes): 15 hours  
- Documentation and submission preparation: 4 hours  

### Limitations and future work
- The dataset is relatively small and was labelled manually, which limits its generalisation. In future, collecting more orthomosaics from different plots, seasons and imaging conditions would enable a more robust model to be trained.
- There is noticeable class imbalance between species, which affects per‑class IoU; future work could apply class‑weighted losses or targeted data collection.  
- Currently, the model operates on fixed-size tiles and does not perform full-scene inference on the complete orthomosaic. The next step is to apply the trained model to the entire orthomosaic using a sliding window/fully convolutional approach, followed by post-processing to merge crowns and evaluate the results at stand level. If time allows, this extension will also be included in the final report.

