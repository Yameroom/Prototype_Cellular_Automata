
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

dem_path = r"D:\Documents\Data for FInal Project Chat GPT Access\Qgis Data\DEMNAS_Pujon_UTM49S.tif"

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(np.float32)
    bounds = src.bounds
    resolution = src.res

valid_mask = np.isfinite(dem)
dem_masked = np.ma.masked_invalid(dem)

light = LightSource(
    azdeg=315,
    altdeg=45
)

hillshade = light.hillshade(
    dem,
    vert_exag=1,
    dx=resolution[0],
    dy=resolution[1]
)

hillshade = np.ma.masked_where(~valid_mask, hillshade)

extent = [
    bounds.left,
    bounds.right,
    bounds.bottom,
    bounds.top
]

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

im1 = axes[0].imshow(
    dem_masked,
    cmap="terrain",
    extent=extent
)

axes[0].set_title("DEMNAS Pujon - Elevation")
axes[0].set_xlabel("UTM Easting (m)")
axes[0].set_ylabel("UTM Northing (m)")
fig.colorbar(im1, ax=axes[0], label="Elevation (m)")

axes[1].imshow(
    hillshade,
    cmap="gray",
    extent=extent
)

axes[1].set_title("DEMNAS Pujon - Hillshade")
axes[1].set_xlabel("UTM Easting (m)")
axes[1].set_ylabel("UTM Northing (m)")

plt.tight_layout()
plt.show()
