# 12507345_Multi-class-segmentation-of-tree-species
The project mainly focuses on distinguishing multiple tree species from UAV orthophotos using a convolutional neural network (CNN) architecture, specifically the U-Net model.

## Exercise 2 – Hacking

### What I implemented
I downloaded high‑resolution orthomosaic data from Zenodo(https://zenodo.org/records/15168163) and manually created tree crown polygons for multiple tree species using QGIS (https://qgis.org/project/overview/). This required learning QGIS , resolving shapefile issues, and carefully delineating individual tree crowns per species.  I used the polygons available from the data itself for some spruce trees and drew others myself.

Based on these , I implemented a complete data set pipeline that (see dataset_pipeline folder):-
- Tiles the orthomosaic into fixed-size image patches,    
- rasterises species-specific crown polygons into semantic segmentation masks, and  
- automatically splits the data into training, validation, and test sets.  

Based on this dataset, I then implemented a separate model training pipeline in PyTorch (see model_training_pipeline folder). This pipeline includes a small U-Net–based semantic segmentation model, data loading , a full training and validation loop, evaluation on a held-out test set, computation of IoU and Dice metrics, and visualization of qualitative predictions.

### Exploration and experiments
The initial setup included six classes (including locust). However, several classes achieved a Mean IoU well below the target threshold of 0.40, even after experimenting with data augmentation using Albumentations.  

To improve robustness and performance, the setup was simplified to five classes (Background (class 0)  Beech (class 1)  Pine (class 2)  Birch (class 3)  Spruce (class 4)) and the dataset was regenerated accordingly. This reduced class confusion and led to improved and more stable Mean IoU and Dice scores, which are reported in the final results (`results/metrics.txt`).  

- With 6 classes and no augmentation, the Mean IoU on the test set was approximately 0.36, clearly below the 0.40 target.  
- After adding data augmentation with Albumentations, the Mean IoU remained at a similar level and did not significantly improve.  
- After reducing the task to 5 classes and regenerating the dataset, the Mean IoU improved to … (final result reported in `results/metrics.txt`).


### Metrics and evaluation
- **Error metrics:** Mean Intersection over Union (IoU) per class and Mean Dice coefficient per class on the test set.  
- **Evaluation protocol:** Metrics computed on a held‑out test set that was never used for training or validation.  
- **Target:** Mean IoU ≥ 0.40.  

**Achieved (test set, averaged over all classes and tiles):**  
- Mean IoU = …  
- Mean Dice = …  
(see `results/metrics.txt` for per‑image and per‑class values).  

### Qualitative results
Qualitative predictions were generated for the test set. For selected test tiles, the following are visualized side‑by‑side:  
- input RGB image tile,  
- ground‑truth semantic mask,  
- predicted semantic mask.  

Each prediction uses a consistent color legend mapping mask colors to tree species. Visual inspection shows that the model learns coherent crown shapes and spatial boundaries, with remaining errors mainly caused by visually similar species.  

### Training behavior
The training and validation loss curves (`results/loss_curve.png`) show stable convergence and no severe overfitting, indicating that the chosen model capacity and training setup are appropriate for the dataset size.  

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
- All paths and file names in the code correspond to the dataset and result files referenced above.  

### Time spent
- Data preparation (data download, QGIS labeling, fixing shapefile issues): 15-16 hours  
- Dataset pipeline (tiling, mask generation, splits, debugging): 8 hours  
- Model and training pipeline (implementation, experiments, tuning, reducing to five classes): 15 hours  
- Documentation and submission preparation: 4 hours  

### Limitations and future work
- The dataset is relatively small and was labelled manually, which limits its generalisation. In future, collecting more orthomosaics from different plots, seasons and imaging conditions would enable a more robust model to be trained.
- There is a class imbalance between tree species, which likely affects per-class IoU. This could be mitigated in future by collecting targeted data for under-represented species and using class-balancing techniques such as weighted losses or oversampling.  
- Currently, the model operates on fixed-size tiles and does not perform full-scene inference on the complete orthomosaic. The next step is to apply the trained model to the entire orthomosaic using a sliding window/fully convolutional approach, followed by post-processing to merge crowns and evaluate the results at stand level. If time allows, this extension will also be included in the final report.

