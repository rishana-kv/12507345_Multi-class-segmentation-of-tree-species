# 12507345_Multi-class-segmentation-of-tree-species
The project mainly focuses on distinguishing multiple tree species from UAV orthophotos using a convolutional neural network (CNN) architecture, specifically the U-Net model.

## Exercise 2 – Hacking

### What I implemented
- I downloaded the orthomosaic data from Zenodo(https://zenodo.org/records/15168163) and created species polygons in QGIS(https://qgis.org/project/overview/) (beech, pine, birch, spruce). I used the polygons available from the data itself for some spruce trees and drew others myself.
- I built a dataset pipeline that tiles the orthomosaic and rasterises polygon masks into training, validation and test splits in Google colab.
- A small U-Net model and a training pipeline were implemented in PyTorch for the semantic segmentation of tree crowns.

### Metrics
- Error metric: Mean IoU per class (and Mean Dice per class) on the test set.
- Target: Mean IoU ≥ 0.40.
- Achieved: Mean IoU = …, Mean Dice = … (from `results/metrics.txt`).

### Time spent
- Data preparation (download, QGIS, tiling): 16 hours  
- Model & training pipeline implementation: Y hours  
- Experiments / tuning: 5 hours  
- Documentation & cleanup: 2 hours

