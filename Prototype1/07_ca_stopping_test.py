
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

dem_path = r"D:\Documents\Data for FInal Project Chat GPT Access\Qgis Data\DEMNAS_Pujon_UTM49S.tif"
selection_path = r"C:\Data Tugas Akhir\Prototype1\source_selection.npz"

material_per_cell = 1.0
exponent = 1.0
critical_slope = 25.0

if not os.path.exists(selection_path):
    raise FileNotFoundError(
        "File source_selection.npz tidak ditemukan. "
        "Jalankan 03_source_selection.py terlebih dahulu."
    )

selection = np.load(selection_path)

center_row = int(selection["row"])
center_col = int(selection["col"])
source_size = int(selection["source_size"])

row_start = int(selection["row_start"])
row_end = int(selection["row_end"])
col_start = int(selection["col_start"])
col_end = int(selection["col_end"])

with rasterio.open(dem_path) as src:
    dem = src.read(1)
    transform = src.transform
    bounds = src.bounds
    dx, dy = src.res
    height, width = dem.shape

valid_mask = np.isfinite(dem)

source_dem = dem[
    row_start:row_end,
    col_start:col_end
]

if source_dem.shape != (source_size, source_size):
    raise ValueError("Ukuran area sumber tidak sesuai.")

if not np.all(np.isfinite(source_dem)):
    raise ValueError("Area sumber mengandung NoData.")

directions = [
    ("NW", -1, -1, np.sqrt(dx**2 + dy**2)),
    ("N",  -1,  0, dy),
    ("NE", -1,  1, np.sqrt(dx**2 + dy**2)),
    ("W",   0, -1, dx),
    ("E",   0,  1, dx),
    ("SW",  1, -1, np.sqrt(dx**2 + dy**2)),
    ("S",   1,  0, dy),
    ("SE",  1,  1, np.sqrt(dx**2 + dy**2))
]

output_row_start = max(0, row_start - 1)
output_row_end = min(height, row_end + 1)

output_col_start = max(0, col_start - 1)
output_col_end = min(width, col_end + 1)

output_shape = (
    output_row_end - output_row_start,
    output_col_end - output_col_start
)

mobile_next = np.zeros(output_shape, dtype=np.float64)
retained_map = np.zeros(output_shape, dtype=np.float64)

initial_material = source_size**2 * material_per_cell

routed_cells = 0
retained_cells = 0
total_distributed = 0.0
total_retained = 0.0

print("\n===== CA STOPPING TEST =====")
print("Center row:", center_row)
print("Center column:", center_col)
print("Source size:", source_size, "x", source_size)
print("Initial material:", initial_material)
print("Critical slope:", critical_slope, "degrees")
print("MFD exponent:", exponent)

for row in range(row_start, row_end):
    for col in range(col_start, col_end):

        current_elevation = dem[row, col]
        lower_neighbors = []

        for name, dr, dc, distance in directions:

            neighbor_row = row + dr
            neighbor_col = col + dc

            if not (
                0 <= neighbor_row < height and
                0 <= neighbor_col < width
            ):
                continue

            if not valid_mask[neighbor_row, neighbor_col]:
                continue

            neighbor_elevation = dem[
                neighbor_row,
                neighbor_col
            ]

            slope_ratio = (
                current_elevation - neighbor_elevation
            ) / distance

            if slope_ratio <= 0:
                continue

            slope_angle = np.degrees(
                np.arctan(slope_ratio)
            )

            if slope_angle >= critical_slope:

                weight = slope_ratio ** exponent

                lower_neighbors.append(
                    (
                        name,
                        neighbor_row,
                        neighbor_col,
                        slope_angle,
                        weight
                    )
                )

        current_material = material_per_cell

        local_row = row - output_row_start
        local_col = col - output_col_start

        if len(lower_neighbors) == 0:

            retained_map[local_row, local_col] += current_material

            total_retained += current_material
            retained_cells += 1

        else:

            total_weight = sum(
                item[4] for item in lower_neighbors
            )

            distributed = 0.0

            for name, nr, nc, angle, weight in lower_neighbors:

                proportion = weight / total_weight
                material = current_material * proportion

                target_row = nr - output_row_start
                target_col = nc - output_col_start

                mobile_next[target_row, target_col] += material

                distributed += material

            total_distributed += distributed
            routed_cells += 1

total_mobile = np.sum(mobile_next)
total_after = total_mobile + np.sum(retained_map)

print("\n===== DISTRIBUTION SUMMARY =====")
print("Source cells:", source_size**2)
print("Cells that routed material:", routed_cells)
print("Cells that retained material:", retained_cells)
print("Total mobile material:", total_mobile)
print("Total retained material:", total_retained)

print("\n===== CONSERVATION CHECK =====")
print("Initial material:", initial_material)
print("Mobile material:", total_mobile)
print("Retained material:", np.sum(retained_map))
print("Total after iteration:", total_after)

if np.isclose(
    initial_material,
    total_after,
    rtol=1e-8,
    atol=1e-8
):
    print("Status: MATERIAL CONSERVED")
else:
    print("Status: MATERIAL NOT CONSERVED")

print("===============================\n")

output_dem = dem[
    output_row_start:output_row_end,
    output_col_start:output_col_end
]

output_left = bounds.left + output_col_start * dx
output_right = bounds.left + output_col_end * dx

output_top = bounds.top - output_row_start * dy
output_bottom = bounds.top - output_row_end * dy

source_x = bounds.left + col_start * dx
source_y = bounds.top - row_end * dy

source_width = source_size * dx
source_height = source_size * dy

map_extent = [
    output_left,
    output_right,
    output_bottom,
    output_top
]

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 7)
)

for ax in axes:
    ax.imshow(
        np.ma.masked_invalid(output_dem),
        cmap="terrain",
        extent=map_extent,
        origin="upper",
        interpolation="nearest"
    )

    source_patch = Rectangle(
        (source_x, source_y),
        source_width,
        source_height,
        linewidth=2,
        edgecolor="blue",
        facecolor="none"
    )

    ax.add_patch(source_patch)

    ax.set_xlabel("UTM Easting (m)")
    ax.set_aspect("equal")

axes[0].set_title("Initial Source Area")
axes[0].set_ylabel("UTM Northing (m)")

axes[1].set_title("Mobile Material")

mobile_display = np.ma.masked_where(
    mobile_next <= 0,
    mobile_next
)

if np.any(mobile_next > 0):

    mobile_image = axes[1].imshow(
        mobile_display,
        cmap="YlOrRd",
        alpha=0.85,
        extent=map_extent,
        origin="upper",
        interpolation="nearest"
    )

    fig.colorbar(
        mobile_image,
        ax=axes[1],
        label="Relative Mobile Material"
    )

axes[2].set_title("Retained Material")

retained_display = np.ma.masked_where(
    retained_map <= 0,
    retained_map
)

if np.any(retained_map > 0):

    retained_image = axes[2].imshow(
        retained_display,
        cmap="Blues",
        alpha=0.85,
        extent=map_extent,
        origin="upper",
        interpolation="nearest"
    )

    fig.colorbar(
        retained_image,
        ax=axes[2],
        label="Relative Retained Material"
    )

fig.suptitle(
    f"CA Stopping Test - Critical Slope {critical_slope} Degrees",
    fontsize=14
)

plt.tight_layout()
plt.show()
