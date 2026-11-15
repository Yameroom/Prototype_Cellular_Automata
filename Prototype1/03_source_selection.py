
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

dem_path = r"D:\Documents\Data for FInal Project Chat GPT Access\Qgis Data\DEMNAS_Pujon_UTM49S.tif"

output_folder = r"C:\Data Tugas Akhir\Prototype1"
selection_path = os.path.join(output_folder, "source_selection.npz")

source_size = 5
half_size = source_size // 2
zoom_radius = 5

with rasterio.open(dem_path) as src:
    dem = src.read(1)
    transform = src.transform
    bounds = src.bounds
    resolution = src.res

    valid_mask = np.isfinite(dem)

    crs = src.crs
    height, width = dem.shape

    dx, dy = resolution

    extent = [
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ]

fig, (ax_map, ax_zoom) = plt.subplots(
    1, 2,
    figsize=(15, 9),
    gridspec_kw={"width_ratios": [1.2, 1]}
)

image = ax_map.imshow(
    np.ma.masked_invalid(dem),
    cmap="terrain",
    extent=extent,
    origin="upper",
    interpolation="nearest"
)

fig.colorbar(
    image,
    ax=ax_map,
    label="Elevation (m)",
    shrink=0.8
)

ax_map.set_title("DEMNAS Pujon - Source Selection")
ax_map.set_xlabel("UTM Easting (m)")
ax_map.set_ylabel("UTM Northing (m)")

ax_map.text(
    0.02,
    0.98,
    "Click to select source center",
    transform=ax_map.transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8)
)

ax_zoom.set_title("Source Area - Zoom View")
ax_zoom.text(
    0.5,
    0.5,
    "Select a point on the map",
    transform=ax_zoom.transAxes,
    ha="center",
    va="center"
)

ax_zoom.set_xlabel("UTM Easting (m)")
ax_zoom.set_ylabel("")

source_patch = None
source_marker = None


def select_source(event):
    global source_patch, source_marker

    if event.inaxes != ax_map:
        return

    if event.xdata is None or event.ydata is None:
        return

    x = event.xdata
    y = event.ydata

    row, col = rasterio.transform.rowcol(
        transform,
        x,
        y
    )

    if not (0 <= row < height and 0 <= col < width):
        print("Lokasi berada di luar raster.")
        return

    if not valid_mask[row, col]:
        print("Lokasi yang dipilih merupakan NoData.")
        return

    row_start = row - half_size
    row_end = row + half_size + 1

    col_start = col - half_size
    col_end = col + half_size + 1

    if (
        row_start < 0 or
        col_start < 0 or
        row_end > height or
        col_end > width
    ):
        print("Source area melewati batas raster.")
        return

    source_mask = valid_mask[
        row_start:row_end,
        col_start:col_end
    ]

    if not np.all(source_mask):
        print("Source area mengenai NoData.")
        print("Silakan pilih lokasi lain.")
        return

    source_elevation = dem[
        row_start:row_end,
        col_start:col_end
    ]

    center_elevation = dem[row, col]

    center_x, center_y = rasterio.transform.xy(
        transform,
        row,
        col
    )

    source_x = bounds.left + col_start * dx
    source_y = bounds.top - row_end * dy

    source_width = source_size * dx
    source_height = source_size * dy

    source_area = source_width * source_height

    # Memperbarui penanda pada peta utama

    if source_patch is not None:
        source_patch.remove()

    if source_marker is not None:
        source_marker.remove()

    source_patch = Rectangle(
        (source_x, source_y),
        source_width,
        source_height,
        linewidth=2,
        edgecolor="red",
        facecolor="red",
        alpha=0.35
    )

    ax_map.add_patch(source_patch)

    source_marker = ax_map.scatter(
        center_x,
        center_y,
        color="red",
        edgecolor="white",
        s=45,
        zorder=5
    )

    # Mengambil area pembesaran

    zoom_row_start = max(0, row - zoom_radius)
    zoom_row_end = min(height, row + zoom_radius + 1)

    zoom_col_start = max(0, col - zoom_radius)
    zoom_col_end = min(width, col + zoom_radius + 1)

    zoom_dem = dem[
        zoom_row_start:zoom_row_end,
        zoom_col_start:zoom_col_end
    ]

    zoom_left = bounds.left + zoom_col_start * dx
    zoom_right = bounds.left + zoom_col_end * dx

    zoom_top = bounds.top - zoom_row_start * dy
    zoom_bottom = bounds.top - zoom_row_end * dy

    # Memperbarui panel pembesaran

    ax_zoom.clear()

    ax_zoom.imshow(
        np.ma.masked_invalid(zoom_dem),
        cmap="terrain",
        vmin=image.get_clim()[0],
        vmax=image.get_clim()[1],
        extent=[
            zoom_left,
            zoom_right,
            zoom_bottom,
            zoom_top
        ],
        origin="upper",
        interpolation="nearest"
    )

    # Menggambar kotak area sumber

    zoom_patch = Rectangle(
        (source_x, source_y),
        source_width,
        source_height,
        linewidth=2,
        edgecolor="red",
        facecolor="red",
        alpha=0.25
    )

    ax_zoom.add_patch(zoom_patch)

    # Menggambar batas setiap sel sumber

    for i in range(source_size + 1):

        grid_x = source_x + i * dx
        grid_y = source_y + i * dy

        ax_zoom.vlines(
            grid_x,
            source_y,
            source_y + source_height,
            color="red",
            linewidth=0.8
        )

        ax_zoom.hlines(
            grid_y,
            source_x,
            source_x + source_width,
            color="red",
            linewidth=0.8
        )

    # Menampilkan titik pusat sumber

    ax_zoom.scatter(
        center_x,
        center_y,
        color="red",
        edgecolor="white",
        s=55,
        zorder=5
    )

    ax_zoom.set_title(
        f"Source Area - {source_size} x {source_size} Cells"
    )

    ax_zoom.set_xlabel("UTM Easting (m)")
    ax_zoom.set_ylabel("")
    ax_zoom.set_aspect("equal")

    fig.canvas.draw_idle()

    # Menyimpan lokasi sumber

    np.savez(
        selection_path,
        row=row,
        col=col,
        source_size=source_size,
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        center_x=center_x,
        center_y=center_y,
        center_elevation=center_elevation
    )

    # Menampilkan informasi area sumber

    print("\n===== SOURCE AREA SELECTED =====")
    print("CRS:", crs)
    print("Center row:", row)
    print("Center column:", col)
    print("Center elevation:", center_elevation, "m")
    print("Center UTM:", center_x, center_y)

    print("Source size:", source_size, "x", source_size)
    print("Number of cells:", source_size ** 2)

    print("Source elevation minimum:", np.min(source_elevation))
    print("Source elevation maximum:", np.max(source_elevation))
    print("Source elevation mean:", np.mean(source_elevation))

    print("Source width:", source_width, "m")
    print("Source height:", source_height, "m")
    print("Source area:", source_area, "m2")

    print("Selection saved to:", selection_path)
    print("===============================\n")


fig.canvas.mpl_connect(
    "button_press_event",
    select_source
)

plt.tight_layout()
plt.show()
