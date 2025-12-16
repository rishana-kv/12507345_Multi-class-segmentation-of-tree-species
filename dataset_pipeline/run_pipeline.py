
import random
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
from shapely.geometry import box
#importing other functions from above
from config import *
from io_utils import ensure_dirs, load_shapefiles
from tiling import extract_tile, rasterize_mask

random.seed(RANDOM_SEED)

def main():
    # Make sure all folders exist
    ensure_dirs(DATASET_DIR)
    # Load all shapefiles
    species_gdfs = load_shapefiles(SHAPEFILES)
    # Open orthomosaic
    with rasterio.open(ORTHO_IMAGE) as img:
        # Convert shapefiles to same CRS as image
        species_gdfs = {k: gdf.to_crs(img.crs) for k, gdf in species_gdfs.items()}
        tile_id = 0
        for gdf in species_gdfs.values():
            for geom in tqdm(gdf.geometry):
                if geom is None or geom.is_empty:
                    continue
                # Extract tile
                tile, transform, tile_box = extract_tile_clamped(img, geom.centroid.x, geom.centroid.y)
                if tile is None:
                    print(f"Skipped tile at centroid: ({geom.centroid.x}, {geom.centroid.y})")
                    continue
                # Create mask
                mask = rasterize_mask(tile_box, transform, tile.shape[1:], species_gdfs, CLASS_MAP)
                # Decide split
                split = random.choices(
                    ["train", "val", "test"],
                    weights=[0.7, 0.15, 0.15]
                )[0]
                # Save tile and mask
                img_path = f"{DATASET_DIR}/{split}/img/{tile_id}.png"
                mask_path = f"{DATASET_DIR}/{split}/mask/{tile_id}.png"

                Image.fromarray(np.moveaxis(tile, 0, -1).astype(np.uint8)).save(img_path)
                Image.fromarray(mask).save(mask_path)

                print(f"Saved tile {tile_id} to {split}")  # Debug info

                tile_id += 1
    print("Dataset generation finished.")


def extract_tile_clamped(img, cx, cy):
    """
    Extracts a tile centered at (cx, cy) but clamps coordinates to image bounds.
    Returns (tile, transform, tile_box) or (None, None, None) if tile too small.
    """
    half = HALF_TILE
    # Convert image bounds to coordinates
    left, bottom = img.transform * (0, img.height)
    right, top = img.transform * (img.width, 0)

    xmin = max(cx - half, left)
    ymin = max(cy - half, bottom)
    xmax = min(cx + half, right)
    ymax = min(cy + half, top)

    window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, img.transform)
    tile = img.read(window=window)

    # Skip tiles smaller than TILE_SIZE
    if tile.shape[1] < TILE_SIZE or tile.shape[2] < TILE_SIZE:
        return None, None, None

    transform = img.window_transform(window)
    return tile[:3], transform, box(xmin, ymin, xmax, ymax)


if __name__ == "__main__":
    main()
