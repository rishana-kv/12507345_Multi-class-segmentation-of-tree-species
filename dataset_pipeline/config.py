
#tiles
TILE_SIZE = 64
HALF_TILE = TILE_SIZE // 2
#to take same data
RANDOM_SEED = 42
# different species
CLASS_MAP = {
    "beech": 1,
    "pine": 2,
    "birch": 3,
    "spruce": 4,
}
# Input data
ORTHO_IMAGE = "/content/drive/MyDrive/All_data/Orthomosaics/orthomosaic.tif"
#polygons for all species

SHAPEFILES = {
    "beech":  "/content/drive/MyDrive/All_data/Crown_shapes/Beech_Crown/Id.shp",
    "pine":   "/content/drive/MyDrive/All_data/Crown_shapes/Pine_Crown/GKI_Pine_Crown.shp",
    "birch":  "/content/drive/MyDrive/All_data/Crown_shapes/Brich_Crown/Crown_Brich.shp",
    "spruce": "/content/drive/MyDrive/All_data/Crown_shapes/Spruce_Crown/Spruce.shp",
}
# Output
DATASET_DIR = "/content/drive/MyDrive/All_data/datasets"
