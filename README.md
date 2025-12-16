# 12507345_Multi-class-segmentation-of-tree-species
The project mainly focuses on distinguishing multiple tree species from UAV orthophotos using a convolutional neural network (CNN) architecture, specifically the U-Net model.

## Exercise 2 – Hacking

### What I implemented
I downloaded high‑resolution orthomosaic data from Zenodo and manually created tree crown polygons for multiple tree species using QGIS. This required learning QGIS, resolving shapefile issues, and carefully delineating individual tree crowns per species.  

Based on these annotations, I implemented a complete dataset pipeline that  
- tiles the orthomosaic into fixed‑size image patches,  
- rasterizes species‑specific crown polygons into semantic segmentation masks, and  
- automatically splits the data into training, validation, and test sets.  

On top of this dataset, I implemented a small U‑Net–based semantic segmentation model in PyTorch. The training pipeline includes data loading, training, validation, testing, metric computation, and visualization of qualitative predictions.  

### Short project story
The initial setup included six tree species (including locust). However, several classes achieved a Mean IoU well below the target threshold of 0.40, even after experimenting with data augmentation using Albumentations.  

To improve robustness and performance, the setup was simplified to five species and the dataset was regenerated accordingly. This reduced class confusion and led to improved and more stable Mean IoU and Dice scores, which are reported in the final results (`results/metrics.txt`).  

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
- Data preparation (data download, QGIS labeling, fixing shapefile issues): X hours  
- Dataset pipeline (tiling, mask generation, splits, debugging): Y hours  
- Model and training pipeline (implementation, experiments, tuning, reducing to five species): Z hours  
- Documentation, repository cleanup, and submission preparation: W hours  

### Limitations
- The dataset is relatively small and manually labeled, which limits generalization.  
- Class imbalance between tree species likely affects per‑class IoU.  
- The model operates on fixed‑size tiles and does not perform full‑scene inference on the complete orthomosaic.  

