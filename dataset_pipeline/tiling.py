


import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import box
#a square tile from orthomosaic
def extract_tile(img, cx, cy, half_size, tile_size):
    xmin, ymin = cx - half_size, cy - half_size
    xmax, ymax = cx + half_size, cy + half_size
    window = rasterio.windows.from_bounds(
        xmin, ymin, xmax, ymax, img.transform
    )
    tile = img.read(window=window)
    if tile.shape[1:] != (tile_size, tile_size):
        return None, None, None
    transform = img.window_transform(window)
    return tile[:3], transform, box(xmin, ymin, xmax, ymax)

#mask function for each species
def rasterize_mask(tile_box, transform, shape, gdfs, class_map):
    mask = np.zeros(shape, dtype=np.uint8)

    for name, class_id in class_map.items():
        geoms = gdfs[name][gdfs[name].geometry.intersects(tile_box)].geometry
        if geoms.empty:
            continue
        layer = rasterio.features.rasterize(
            [(g, class_id) for g in geoms],
            out_shape=shape,
            transform=transform,
            fill=0
        )
        mask[(mask == 0) & (layer > 0)] = layer[(mask == 0) & (layer > 0)]
    return mask
