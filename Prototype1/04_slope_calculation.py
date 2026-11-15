
import rasterio
import numpy as np
import matplotlib.pyplot as plt

dem_path = r"D:\Documents\Data for FInal Project Chat GPT Access\Qgis Data\DEMNAS_Pujon_UTM49S.tif"

output_path = r"C:\Data Tugas Akhir\Prototype1\Slope_Pujon_CA.tif"

with rasterio.open(dem_path) as src:

    dem = src.read(1)

    transform = src.transform
    bounds = src.bounds
    resolution = src.res

    profile = src.profile.copy()

    crs = src.crs
    height, width = dem.shape

    dx, dy = resolution

    extent = [
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ]

    print("===== DEM INFORMATION =====")
    print("CRS:", crs)
    print("Raster size:", width, "x", height)
    print("Resolution X:", dx, "m")
    print("Resolution Y:", dy, "m")
    print("===========================\n")

    # Menghitung perubahan elevasi

    dz_dy, dz_dx = np.gradient(
        dem.astype(np.float64),
        dy,
        dx
    )

    # Menghitung besar kemiringan lereng

    slope = np.degrees(
        np.arctan(
            np.sqrt(
                dz_dx ** 2 + dz_dy ** 2
            )
        )
    )

    # Mempertahankan area NoData

    valid_mask = np.isfinite(dem)
    slope[~valid_mask] = np.nan

    # Mengubah hasil menjadi Float32

    slope_output = np.where(
        np.isfinite(slope),
        slope,
        -9999
    ).astype(np.float32)

    # Menyiapkan metadata GeoTIFF

    profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=-9999,
        compress="lzw"
    )

    # Menyimpan hasil slope

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(slope_output, 1)

    # Statistik hasil slope

    valid_slope = slope[np.isfinite(slope)]

    print("===== SLOPE STATISTICS =====")
    print("Minimum slope:", np.min(valid_slope), "degrees")
    print("Maximum slope:", np.max(valid_slope), "degrees")
    print("Mean slope:", np.mean(valid_slope), "degrees")
    print("Median slope:", np.median(valid_slope), "degrees")
    print("Standard deviation:", np.std(valid_slope), "degrees")
    print("Valid cells:", len(valid_slope))
    print("============================\n")

    print("Slope raster berhasil disimpan:")
    print(output_path)

# Menampilkan DEM dan slope

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(15, 8)
)

dem_image = ax1.imshow(
    np.ma.masked_invalid(dem),
    cmap="terrain",
    extent=extent,
    origin="upper"
)

fig.colorbar(
    dem_image,
    ax=ax1,
    label="Elevation (m)",
    shrink=0.8
)

ax1.set_title("DEMNAS Pujon")
ax1.set_xlabel("UTM Easting (m)")
ax1.set_ylabel("UTM Northing (m)")

slope_image = ax2.imshow(
    np.ma.masked_invalid(slope),
    cmap="viridis",
    extent=extent,
    origin="upper",
    vmin=0
)

fig.colorbar(
    slope_image,
    ax=ax2,
    label="Slope (degrees)",
    shrink=0.8
)

ax2.set_title("Slope Pujon")
ax2.set_xlabel("UTM Easting (m)")
ax2.set_ylabel("UTM Northing (m)")

plt.tight_layout()
plt.show()
