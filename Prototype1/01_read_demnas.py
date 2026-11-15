
import rasterio
import numpy as np
import matplotlib.pyplot as plt

dem_path = r"D:\Documents\Data for FInal Project Chat GPT Access\Qgis Data\DEMNAS_Pujon_UTM49S.tif"

with rasterio.open(dem_path) as src:
    dem = src.read(1)

    print("===== DEMNAS METADATA =====")
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)
    print("Data type:", src.dtypes[0])
    print("NoData:", src.nodata)

    valid = dem[np.isfinite(dem)]

    print("\n===== ELEVATION =====")
    print("Minimum:", np.min(valid))
    print("Maximum:", np.max(valid))
    print("Mean:", np.mean(valid))
    print("Valid cells:", len(valid))
    print("Total cells:", dem.size)
    print("Valid percentage:", len(valid) / dem.size * 100)

    plt.figure(figsize=(9, 9))
    plt.imshow(dem, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title("DEMNAS Pujon")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()
