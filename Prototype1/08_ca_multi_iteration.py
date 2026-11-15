
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from rasterio.windows import Window
from rasterio.transform import array_bounds
from matplotlib.colors import LightSource

dem_path = r"D:\Documents\Data for Final Project Chat GPT Access\Qgis Data\DEMNAS_Pujon_UTM49S.tif"
source_path = r"C:\Data Tugas Akhir\Prototype1\source_selection.npz"

material_per_cell = 1.0
exponent = 1.0
critical_slope = 15.0

window_radius = 150
max_iterations = 100
epsilon = 1e-12

directions = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

# ==========================================
# 1. MEMBACA LOKASI AREA SUMBER
# ==========================================

with np.load(source_path) as source_data:
    center_row = int(source_data["row"])
    center_col = int(source_data["col"])
    source_size = int(source_data["source_size"])

half_size = source_size // 2

source_row_start = center_row - half_size
source_row_end = source_row_start + source_size

source_col_start = center_col - half_size
source_col_end = source_col_start + source_size

# ==========================================
# 2. MEMBACA DEM DAN WINDOW SIMULASI
# ==========================================

with rasterio.open(dem_path) as src:

    dem_height = src.height
    dem_width = src.width

    row_min = max(0, center_row - window_radius)
    row_max = min(dem_height, center_row + window_radius + 1)

    col_min = max(0, center_col - window_radius)
    col_max = min(dem_width, center_col + window_radius + 1)

    window = Window(
        col_min,
        row_min,
        col_max - col_min,
        row_max - row_min
    )

    dem_masked = src.read(
        1,
        window=window,
        masked=True
    )

    dem = dem_masked.astype(np.float64).filled(np.nan)

    transform = src.window_transform(window)

    dx = abs(transform.a)
    dy = abs(transform.e)

    crs = src.crs

    print("===== DEM INFORMATION =====")
    print("CRS:", crs)
    print("DEM window:", dem.shape)
    print("Cell size X:", dx)
    print("Cell size Y:", dy)

    if crs is None or not crs.is_projected:
        raise ValueError("DEM harus menggunakan CRS proyeksi.")

    if transform.b != 0 or transform.d != 0:
        raise ValueError("Program ini mengasumsikan raster north-up.")

# ==========================================
# 3. MENYESUAIKAN KOORDINAT SUMBER
# ==========================================

local_source_row_start = source_row_start - row_min
local_source_row_end = source_row_end - row_min

local_source_col_start = source_col_start - col_min
local_source_col_end = source_col_end - col_min

if (
    local_source_row_start < 0
    or local_source_col_start < 0
    or local_source_row_end > dem.shape[0]
    or local_source_col_end > dem.shape[1]
):
    raise ValueError("Source area berada di luar window simulasi.")

source_dem = dem[
    local_source_row_start:local_source_row_end,
    local_source_col_start:local_source_col_end
]

if source_dem.shape != (source_size, source_size):
    raise ValueError("Ukuran source area tidak sesuai.")

if not np.all(np.isfinite(source_dem)):
    raise ValueError("Source area mengandung sel NoData.")

# ==========================================
# 4. INISIALISASI MATERIAL
# ==========================================

mobile_current = np.zeros(dem.shape, dtype=np.float64)

mobile_current[
    local_source_row_start:local_source_row_end,
    local_source_col_start:local_source_col_end
] = material_per_cell

retained_total = np.zeros(dem.shape, dtype=np.float64)

source_mask = np.zeros(dem.shape, dtype=bool)

source_mask[
    local_source_row_start:local_source_row_end,
    local_source_col_start:local_source_col_end
] = True

runout_footprint = source_mask.copy()

initial_material = np.sum(mobile_current)

frames = []

frames.append((
    mobile_current.astype(np.float32).copy(),
    runout_footprint.copy(),
    retained_total.astype(np.float32).copy()
))

iteration_results = []

stop_reason = "Maximum iteration limit reached."

# ==========================================
# 5. INFORMASI AWAL SIMULASI
# ==========================================

print("\n===== CA MULTI-ITERATION =====")
print("Center row:", center_row)
print("Center column:", center_col)
print("Source size:", source_size, "x", source_size)
print("Initial material:", initial_material)
print("Critical slope:", critical_slope, "degrees")
print("MFD exponent:", exponent)

# ==========================================
# 6. PROSES CELLULAR AUTOMATA
# ==========================================

for iteration in range(1, max_iterations + 1):

    mobile_next = np.zeros(dem.shape, dtype=np.float64)

    active_cells = np.argwhere(mobile_current > epsilon)

    routed_cells = 0
    stopped_cells = 0

    for row, col in active_cells:

        current_material = mobile_current[row, col]
        current_elevation = dem[row, col]

        eligible_neighbors = []
        weights = []

        for dr, dc in directions:

            neighbor_row = row + dr
            neighbor_col = col + dc

            if not (
                0 <= neighbor_row < dem.shape[0]
                and 0 <= neighbor_col < dem.shape[1]
            ):
                continue

            neighbor_elevation = dem[
                neighbor_row,
                neighbor_col
            ]

            if not np.isfinite(neighbor_elevation):
                continue

            elevation_difference = (
                current_elevation - neighbor_elevation
            )

            if elevation_difference <= 0:
                continue

            distance = np.sqrt(
                (dc * dx) ** 2 +
                (dr * dy) ** 2
            )

            slope_ratio = elevation_difference / distance

            slope_angle = np.degrees(
                np.arctan(slope_ratio)
            )

            if slope_angle >= critical_slope:

                weight = slope_ratio ** exponent

                eligible_neighbors.append(
                    (neighbor_row, neighbor_col)
                )

                weights.append(weight)

        if len(eligible_neighbors) == 0:

            retained_total[row, col] += current_material

            stopped_cells += 1

        else:

            weights = np.array(
                weights,
                dtype=np.float64
            )

            proportions = weights / np.sum(weights)

            distributed_material = (
                current_material * proportions
            )

            if len(distributed_material) > 1:

                distributed_material[-1] = (
                    current_material -
                    np.sum(distributed_material[:-1])
                )

            for (neighbor_row, neighbor_col), amount in zip(
                eligible_neighbors,
                distributed_material
            ):

                mobile_next[
                    neighbor_row,
                    neighbor_col
                ] += amount

                if amount > epsilon:

                    runout_footprint[
                        neighbor_row,
                        neighbor_col
                    ] = True

            routed_cells += 1

    # ======================================
    # 7. PEMERIKSAAN KONSERVASI MATERIAL
    # ======================================

    total_mobile = np.sum(mobile_next)
    total_retained = np.sum(retained_total)

    total_after = total_mobile + total_retained

    conserved = np.isclose(
        initial_material,
        total_after,
        rtol=1e-8,
        atol=1e-8
    )

    if not conserved:

        raise RuntimeError(
            f"Material tidak terkonservasi pada iterasi {iteration}. "
            f"Total awal: {initial_material}, "
            f"total akhir: {total_after}"
        )

    iteration_results.append({
        "iteration": iteration,
        "active": len(active_cells),
        "routed": routed_cells,
        "stopped": stopped_cells,
        "mobile": total_mobile,
        "retained": total_retained,
        "conserved": conserved
    })

    print(
        f"Iteration {iteration:03d} | "
        f"Active: {len(active_cells):5d} | "
        f"Routed: {routed_cells:5d} | "
        f"Stopped: {stopped_cells:5d} | "
        f"Mobile: {total_mobile:.6f} | "
        f"Retained: {total_retained:.6f}"
    )

    mobile_current = mobile_next

    frames.append((
        mobile_current.astype(np.float32).copy(),
        runout_footprint.copy(),
        retained_total.astype(np.float32).copy()
    ))

    # ======================================
    # 8. KONDISI PENGHENTIAN SIMULASI
    # ======================================

    if total_mobile <= epsilon:

        stop_reason = "No mobile material remains."

        print("\nSIMULATION STOPPED: No mobile material remains.")

        break

    boundary_reached = (
        np.any(mobile_current[0, :] > epsilon)
        or np.any(mobile_current[-1, :] > epsilon)
        or np.any(mobile_current[:, 0] > epsilon)
        or np.any(mobile_current[:, -1] > epsilon)
    )

    if boundary_reached:

        stop_reason = "Material reached simulation window boundary."

        print("\nWARNING: Material reached simulation window boundary.")
        print("Increase window_radius before interpreting the result.")

        break

else:

    print("\nWARNING: Maximum iteration limit reached.")

# ==========================================
# 9. RINGKASAN HASIL SIMULASI
# ==========================================

final_mobile = np.sum(mobile_current)
final_retained = np.sum(retained_total)

final_total = final_mobile + final_retained

footprint_cells = np.count_nonzero(runout_footprint)

footprint_area = footprint_cells * dx * dy

print("\n===== FINAL SUMMARY =====")
print("Iterations completed:", len(iteration_results))
print("Stop reason:", stop_reason)

print("Initial material:", initial_material)
print("Final mobile material:", final_mobile)
print("Final retained material:", final_retained)
print("Total material:", final_total)

print("Runout footprint cells:", footprint_cells)
print("Runout footprint area:", round(footprint_area, 2), "m2")

print("\n===== CONSERVATION CHECK =====")

print(
    "Status:",
    "MATERIAL CONSERVED"
    if np.isclose(
        initial_material,
        final_total,
        rtol=1e-8,
        atol=1e-8
    )
    else "MATERIAL NOT CONSERVED"
)

# ==========================================
# 10. PERSIAPAN HILLSHADE
# ==========================================

valid_dem = np.isfinite(dem)

dem_for_shading = np.where(
    valid_dem,
    dem,
    np.nanmedian(dem)
)

light_source = LightSource(
    azdeg=315,
    altdeg=35
)

hillshade = light_source.hillshade(
    dem_for_shading,
    vert_exag=1.15,
    dx=dx,
    dy=dy
)

valid_shade = hillshade[valid_dem]

low = np.percentile(valid_shade, 1)
high = np.percentile(valid_shade, 99)

hillshade = np.clip(
    (hillshade - low) / (high - low),
    0,
    1
)

hillshade = 0.5 + (hillshade - 0.5) * 0.55
hillshade = hillshade + 0.12

hillshade = np.clip(
    hillshade,
    0,
    1
)

hillshade[~valid_dem] = np.nan

west, south, east, north = array_bounds(
    dem.shape[0],
    dem.shape[1],
    transform
)

extent = [
    west,
    east,
    south,
    north
]

# ==========================================
# 11. MENENTUKAN AREA ZOOM DETAIL
# ==========================================

footprint_positions = np.argwhere(runout_footprint)

if len(footprint_positions) == 0:
    raise ValueError("Runout footprint tidak ditemukan.")

footprint_row_min = np.min(footprint_positions[:, 0])
footprint_row_max = np.max(footprint_positions[:, 0])

footprint_col_min = np.min(footprint_positions[:, 1])
footprint_col_max = np.max(footprint_positions[:, 1])

footprint_height = footprint_row_max - footprint_row_min + 1
footprint_width = footprint_col_max - footprint_col_min + 1

zoom_margin = max(
    10,
    int(max(footprint_height, footprint_width) * 0.20)
)

zoom_row_start = max(
    0,
    footprint_row_min - zoom_margin
)

zoom_row_end = min(
    dem.shape[0],
    footprint_row_max + zoom_margin + 1
)

zoom_col_start = max(
    0,
    footprint_col_min - zoom_margin
)

zoom_col_end = min(
    dem.shape[1],
    footprint_col_max + zoom_margin + 1
)

zoom_extent = [
    transform.c + zoom_col_start * dx,
    transform.c + zoom_col_end * dx,
    transform.f - zoom_row_end * dy,
    transform.f - zoom_row_start * dy
]

zoom_slice = (
    slice(zoom_row_start, zoom_row_end),
    slice(zoom_col_start, zoom_col_end)
)

print("\n===== ZOOM DETAIL INFORMATION =====")
print("Zoom rows:", zoom_row_start, "to", zoom_row_end)
print("Zoom columns:", zoom_col_start, "to", zoom_col_end)
print("Zoom shape:", (
    zoom_row_end - zoom_row_start,
    zoom_col_end - zoom_col_start
))
print("Zoom margin:", zoom_margin, "cells")

# ==========================================
# 12. SKALA VISUALISASI
# ==========================================

max_mobile = max(
    np.max(frame[0]) for frame in frames
)

max_retained = max(
    np.max(frame[2]) for frame in frames
)

if max_mobile <= 0:
    max_mobile = 1.0

if max_retained <= 0:
    max_retained = 1.0

# ==========================================
# 13. MEMBUAT EMPAT PANEL
# ==========================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 12)
)

ax_mobile = axes[0, 0]
ax_footprint = axes[0, 1]
ax_retained = axes[1, 0]
ax_zoom = axes[1, 1]

fig.subplots_adjust(
    top=0.90,
    bottom=0.07,
    wspace=0.28,
    hspace=0.30
)

overview_axes = [
    ax_mobile,
    ax_footprint,
    ax_retained
]

# ==========================================
# 14. LATAR DEM KESELURUHAN AREA
# ==========================================

for ax in overview_axes:

    ax.imshow(
        hillshade,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=extent,
        origin="upper",
        interpolation="nearest"
    )

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    ax.set_aspect("equal")

    ax.grid(
        True,
        alpha=0.2
    )

ax_mobile.set_title("Mobile Material - Overview")
ax_footprint.set_title("Runout Footprint - Overview")
ax_retained.set_title("Retained Material - Overview")

# ==========================================
# 15. LATAR DEM PANEL ZOOM
# ==========================================

ax_zoom.imshow(
    hillshade[zoom_slice],
    cmap="gray",
    vmin=0,
    vmax=1,
    extent=zoom_extent,
    origin="upper",
    interpolation="nearest"
)

ax_zoom.set_xlim(
    zoom_extent[0],
    zoom_extent[1]
)

ax_zoom.set_ylim(
    zoom_extent[2],
    zoom_extent[3]
)

ax_zoom.set_xlabel("UTM Easting (m)")
ax_zoom.set_ylabel("UTM Northing (m)")
ax_zoom.set_aspect("equal")

ax_zoom.grid(
    True,
    alpha=0.2
)

ax_zoom.set_title("Zoom Detail - Footprint & Material")

# ==========================================
# 16. MOBILE MATERIAL OVERVIEW
# ==========================================

mobile_image = ax_mobile.imshow(
    np.ma.masked_where(
        frames[0][0] <= epsilon,
        frames[0][0]
    ),
    cmap="Reds",
    alpha=0.90,
    vmin=0,
    vmax=max_mobile,
    extent=extent,
    origin="upper"
)

# ==========================================
# 17. RUNOUT FOOTPRINT OVERVIEW
# ==========================================

footprint_image = ax_footprint.imshow(
    np.ma.masked_where(
        ~frames[0][1],
        frames[0][1].astype(np.float32)
    ),
    cmap="YlOrBr",
    alpha=0.85,
    vmin=0,
    vmax=1,
    extent=extent,
    origin="upper"
)

# ==========================================
# 18. RETAINED MATERIAL OVERVIEW
# ==========================================

retained_image = ax_retained.imshow(
    np.ma.masked_where(
        frames[0][2] <= epsilon,
        frames[0][2]
    ),
    cmap="Blues",
    alpha=0.90,
    vmin=0,
    vmax=max_retained,
    extent=extent,
    origin="upper"
)

# ==========================================
# 19. TIGA OVERLAY PADA PANEL ZOOM
# ==========================================

zoom_mobile_image = ax_zoom.imshow(
    np.ma.masked_where(
        frames[0][0][zoom_slice] <= epsilon,
        frames[0][0][zoom_slice]
    ),
    cmap="Reds",
    alpha=0.85,
    vmin=0,
    vmax=max_mobile,
    extent=zoom_extent,
    origin="upper"
)

zoom_footprint_image = ax_zoom.imshow(
    np.ma.masked_where(
        ~frames[0][1][zoom_slice],
        frames[0][1][zoom_slice].astype(np.float32)
    ),
    cmap="YlOrBr",
    alpha=0.35,
    vmin=0,
    vmax=1,
    extent=zoom_extent,
    origin="upper"
)

zoom_retained_image = ax_zoom.imshow(
    np.ma.masked_where(
        frames[0][2][zoom_slice] <= epsilon,
        frames[0][2][zoom_slice]
    ),
    cmap="Blues",
    alpha=0.85,
    vmin=0,
    vmax=max_retained,
    extent=zoom_extent,
    origin="upper"
)

# ==========================================
# 20. PENANDA AREA SUMBER
# ==========================================

source_x = transform.c + local_source_col_start * dx

source_y = transform.f - local_source_row_end * dy

source_width = source_size * dx
source_height = source_size * dy

for ax in overview_axes + [ax_zoom]:

    source_patch = Rectangle(
        (source_x, source_y),
        source_width,
        source_height,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        zorder=10
    )

    ax.add_patch(source_patch)

# ==========================================
# 21. COLORBAR
# ==========================================

fig.colorbar(
    mobile_image,
    ax=ax_mobile,
    label="Relative mobile material",
    shrink=0.8
)

fig.colorbar(
    footprint_image,
    ax=ax_footprint,
    label="Visited cells (0 or 1)",
    shrink=0.8
)

fig.colorbar(
    retained_image,
    ax=ax_retained,
    label="Relative retained material",
    shrink=0.8
)

title = fig.suptitle(
    "CA Landslide Runout - Iteration 0",
    fontsize=15
)

# ==========================================
# 22. FUNGSI ANIMASI
# ==========================================

def update(frame_index):

    mobile_data = frames[frame_index][0]
    footprint_data = frames[frame_index][1]
    retained_data = frames[frame_index][2]

    # Overview mobile

    mobile_image.set_data(
        np.ma.masked_where(
            mobile_data <= epsilon,
            mobile_data
        )
    )

    # Overview footprint

    footprint_image.set_data(
        np.ma.masked_where(
            ~footprint_data,
            footprint_data.astype(np.float32)
        )
    )

    # Overview retained

    retained_image.set_data(
        np.ma.masked_where(
            retained_data <= epsilon,
            retained_data
        )
    )

    # Zoom mobile

    zoom_mobile_image.set_data(
        np.ma.masked_where(
            mobile_data[zoom_slice] <= epsilon,
            mobile_data[zoom_slice]
        )
    )

    # Zoom footprint

    zoom_footprint_image.set_data(
        np.ma.masked_where(
            ~footprint_data[zoom_slice],
            footprint_data[zoom_slice].astype(np.float32)
        )
    )

    # Zoom retained

    zoom_retained_image.set_data(
        np.ma.masked_where(
            retained_data[zoom_slice] <= epsilon,
            retained_data[zoom_slice]
        )
    )

    title.set_text(
        f"CA Landslide Runout - Iteration {frame_index}"
    )

    return (
        mobile_image,
        footprint_image,
        retained_image,
        zoom_mobile_image,
        zoom_footprint_image,
        zoom_retained_image,
        title
    )

# ==========================================
# 23. MENJALANKAN ANIMASI
# ==========================================

animation = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=300,
    repeat=False,
    blit=False
)

plt.show()
