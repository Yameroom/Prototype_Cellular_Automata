
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

dem_path = r"D:\Documents\Data for FInal Project Chat GPT Access\Qgis Data\DEMNAS_Pujon_UTM49S.tif"
selection_path = r"C:\Data Tugas Akhir\Prototype1\source_selection.npz"

material_per_cell = 1.0
exponent = 1.0

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

if not np.all(valid_mask[row_start:row_end, col_start:col_end]):
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

next_material = np.zeros(
    (
        output_row_end - output_row_start,
        output_col_end - output_col_start
    ),
    dtype=np.float64
)

initial_material = source_size**2 * material_per_cell

routed_cells = 0
retained_cells = 0
total_distributed = 0.0
total_retained = 0.0

print("\n===== MFD SOURCE AREA TEST =====")
print("Center row:", center_row)
print("Center column:", center_col)
print("Source size:", source_size, "x", source_size)
print("Material per cell:", material_per_cell)
print("Total initial material:", initial_material)
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

            slope = (
                current_elevation - neighbor_elevation
            ) / distance

            if slope > 0:
                weight = slope ** exponent

                lower_neighbors.append(
                    (name, neighbor_row, neighbor_col, slope, weight)
                )

        current_material = material_per_cell

        if len(lower_neighbors) == 0:

            local_row = row - output_row_start
            local_col = col - output_col_start

            next_material[local_row, local_col] += current_material

            total_retained += current_material
            retained_cells += 1

        else:

            total_weight = sum(
                item[4] for item in lower_neighbors
            )

            distributed = 0.0

            for name, nr, nc, slope, weight in lower_neighbors:

                proportion = weight / total_weight
                material = current_material * proportion

                local_row = nr - output_row_start
                local_col = nc - output_col_start

                next_material[local_row, local_col] += material

                distributed += material

            total_distributed += distributed
            routed_cells += 1

total_after = np.sum(next_material)

print("\n===== DISTRIBUTION SUMMARY =====")
print("Source cells:", source_size**2)
print("Cells that routed material:", routed_cells)
print("Cells that retained material:", retained_cells)
print("Total distributed material:", total_distributed)
print("Total retained material:", total_retained)

print("\n===== CONSERVATION CHECK =====")
print("Initial material:", initial_material)
print("Material after iteration:", total_after)

if np.isclose(
    initial_material,
    total_after,
    rtol=1e-9,
    atol=1e-9
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

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(15, 8)
)

ax1.imshow(
    np.ma.masked_invalid(output_dem),
    cmap="terrain",
    extent=[
        output_left,
        output_right,
        output_bottom,
        output_top
    ],
    origin="upper",
    interpolation="nearest"
)

source_patch1 = Rectangle(
    (source_x, source_y),
    source_width,
    source_height,
    linewidth=2,
    edgecolor="red",
    facecolor="red",
    alpha=0.3
)

ax1.add_patch(source_patch1)

ax1.set_title("Initial Source Area")
ax1.set_xlabel("UTM Easting (m)")
ax1.set_ylabel("UTM Northing (m)")
ax1.set_aspect("equal")

material_display = np.ma.masked_where(
    next_material <= 0,
    next_material
)

ax2.imshow(
    np.ma.masked_invalid(output_dem),
    cmap="terrain",
    extent=[
        output_left,
        output_right,
        output_bottom,
        output_top
    ],
    origin="upper",
    interpolation="nearest"
)

material_image = ax2.imshow(
    material_display,
    cmap="YlOrRd",
    alpha=0.85,
    extent=[
        output_left,
        output_right,
        output_bottom,
        output_top
    ],
    origin="upper",
    interpolation="nearest"
)

source_patch2 = Rectangle(
    (source_x, source_y),
    source_width,
    source_height,
    linewidth=2,
    edgecolor="blue",
    facecolor="none"
)

ax2.add_patch(source_patch2)

fig.colorbar(
    material_image,
    ax=ax2,
    label="Relative Material"
)

ax2.set_title("Material Distribution - 1 Iteration")
ax2.set_xlabel("UTM Easting (m)")
ax2.set_ylabel("UTM Northing (m)")
ax2.set_aspect("equal")

fig.suptitle(
    "MFD Distribution from 5 x 5 Source Area",
    fontsize=14
)

plt.tight_layout()
plt.show()
