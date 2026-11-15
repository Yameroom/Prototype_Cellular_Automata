
import rasterio
import numpy as np
import matplotlib.pyplot as plt

dem_path = r"D:\Documents\Data for FInal Project Chat GPT Access\Qgis Data\DEMNAS_Pujon_UTM49S.tif"

row = 1000
col = 800

initial_material = 100.0
exponent = 1.0

with rasterio.open(dem_path) as src:

    dem = src.read(1)

    dx, dy = src.res

    height, width = dem.shape

    center_elevation = dem[row, col]

    if not np.isfinite(center_elevation):
        raise ValueError("Sel pusat merupakan NoData.")

    neighbors = [
        (-1, -1, "NW", np.sqrt(dx**2 + dy**2)),
        (-1,  0, "N", dy),
        (-1,  1, "NE", np.sqrt(dx**2 + dy**2)),
        ( 0, -1, "W", dx),
        ( 0,  1, "E", dx),
        ( 1, -1, "SW", np.sqrt(dx**2 + dy**2)),
        ( 1,  0, "S", dy),
        ( 1,  1, "SE", np.sqrt(dx**2 + dy**2))
    ]

    slopes = []
    directions = []

    print("\n===== NEIGHBOR ELEVATION CHECK =====")

    print("Center elevation:", center_elevation, "m")

    for dr, dc, name, distance in neighbors:

        nr = row + dr
        nc = col + dc

        if not (0 <= nr < height and 0 <= nc < width):
            print(name, ": Outside raster")
            continue

        neighbor_elevation = dem[nr, nc]

        if not np.isfinite(neighbor_elevation):
            print(name, ": NoData")
            continue

        elevation_difference = (
            center_elevation - neighbor_elevation
        )

        if elevation_difference > 0:

            slope = elevation_difference / distance

            slopes.append(slope)
            directions.append(name)

            print(
                name,
                "| Elevation:", round(float(neighbor_elevation), 4),
                "| Slope:", round(float(slope), 4),
                "| Lower terrain"
            )

        else:

            print(
                name,
                "| Elevation:", round(float(neighbor_elevation), 4),
                "| Higher or equal terrain"
            )

    if len(slopes) == 0:
        raise ValueError(
            "Tidak ditemukan tetangga yang lebih rendah."
        )

    slopes = np.array(slopes)

    weights = slopes ** exponent

    total_weight = np.sum(weights)

    proportions = weights / total_weight

    material_distribution = initial_material * proportions

    distributed_material = np.sum(material_distribution)

    retained_material = initial_material - distributed_material

    total_material = distributed_material + retained_material

    # Hasil perhitungan MFD

    print("\n===== MFD TEST RESULTS =====")

    print("Center row:", row)
    print("Center column:", col)
    print("Center elevation:", center_elevation, "m")

    print("Initial material:", initial_material)

    print("\n===== MATERIAL DISTRIBUTION =====")

    for name, slope, proportion, material in zip(
        directions,
        slopes,
        proportions,
        material_distribution
    ):

        print(
            name,
            "| Slope:", round(float(slope), 4),
            "| Proportion:", round(float(proportion * 100), 2), "%",
            "| Material:", round(float(material), 4)
        )

    print("\n===== CONSERVATION CHECK =====")

    print("Initial material:", initial_material)
    print("Distributed material:", distributed_material)
    print("Retained material:", retained_material)
    print("Total material:", total_material)

    if np.isclose(
        initial_material,
        total_material,
        rtol=1e-10,
        atol=1e-10
    ):
        print("Status: MATERIAL CONSERVED")
    else:
        print("Status: MATERIAL NOT CONSERVED")

    print("============================\n")


# Visualisasi distribusi material

distribution_grid = np.zeros((3, 3))

direction_positions = {
    "NW": (0, 0),
    "N":  (0, 1),
    "NE": (0, 2),
    "W":  (1, 0),
    "E":  (1, 2),
    "SW": (2, 0),
    "S":  (2, 1),
    "SE": (2, 2)
}

for name, material in zip(
    directions,
    material_distribution
):

    r, c = direction_positions[name]

    distribution_grid[r, c] = material


fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    distribution_grid,
    cmap="YlOrRd",
    vmin=0,
    vmax=45
)

for r in range(3):

    for c in range(3):

        if r == 1 and c == 1:

            label = "CENTER"

        else:

            label = f"{distribution_grid[r, c]:.2f}"

        ax.text(
            c,
            r,
            label,
            ha="center",
            va="center",
            color="black",
            fontsize=11
        )

ax.set_title("MFD Material Distribution")

ax.set_xticks([])
ax.set_yticks([])

fig.colorbar(
    image,
    ax=ax,
    label="Material Units"
)

plt.tight_layout()
plt.show()
