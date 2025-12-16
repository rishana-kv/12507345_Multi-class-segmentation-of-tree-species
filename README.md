# 12507345_Multi-class-segmentation-of-tree-species
The project mainly focuses on distinguishing multiple tree species from UAV orthophotos using a convolutional neural network (CNN) architecture, specifically the U-Net model.

## Exercise 2 – Hacking

### What I implemented
- Downloaded orthomosaic data from Zenodo and created species polygons in QGIS (beech, pine, birch, spruce).
- Built a dataset pipeline that tiles the orthomosaic and rasterizes polygon masks into train/val/test splits.
- Implemented a small U-Net model and a training pipeline in PyTorch for semantic segmentation of tree crowns.

### Metrics
- Error metric: Mean IoU per class (and Mean Dice per class) on the test set.
- Target: Mean IoU ≥ 0.40.
- Achieved: Mean IoU = …, Mean Dice = … (from `results/metrics.txt`).

### Time spent
- Data preparation (download, QGIS, tiling): X hours  
- Model & training pipeline implementation: Y hours  
- Experiments / tuning: Z hours  
- Documentation & cleanup: W hours

