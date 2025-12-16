
import os
import geopandas as gpd
#spliting into training , validation and testing datasets
def ensure_dirs(base_dir):
    for split in ["train", "val", "test"]:
        os.makedirs(f"{base_dir}/{split}/img", exist_ok=True)
        os.makedirs(f"{base_dir}/{split}/mask", exist_ok=True)
#reading the shapefiles
def load_shapefiles(shapefile_dict):
    return {k: gpd.read_file(v) for k, v in shapefile_dict.items()}
