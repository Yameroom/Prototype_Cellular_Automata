
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

rows = 200
cols = 200

rng = np.random.default_rng(42)

y, x = np.mgrid[0:rows, 0:cols]

base_slope = 220 - 0.65 * y

terrain_variation = (
    0.8 * np.sin(x / 15)
    + 0.6 * np.sin(y / 18)
    + rng.normal(0, 0.2, (rows, cols))
)

valley_center = 100 + 0.12 * y

terrains = {
    "Lereng Lurus": base_slope + terrain_variation,
    "Lembah": (
        base_slope
        - 22 * np.exp(-((x - valley_center) ** 2) / 350)
        + terrain_variation
    ),
    "Punggung Bukit": (
        base_slope
        + 22 * np.exp(-((x - valley_center) ** 2) / 500)
        + terrain_variation
    ),
    "Lereng Landai": (
        180 - 0.20 * y
        + 0.5 * np.sin(x / 20)
        + terrain_variation * 0.3
    )
}

susceptibility = rng.random((rows, cols))

for _ in range(12):
    padded = np.pad(susceptibility, 1, mode="edge")
    susceptibility = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0

susceptibility = (
    (susceptibility - susceptibility.min())
    / (susceptibility.max() - susceptibility.min())
)

source_row = 25
source_col = 100
susceptibility_threshold = 0.20
max_iterations = 300

neighbors = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

results = {}

for name, elevation in terrains.items():
    landslide = np.zeros((rows, cols), dtype=np.uint8)

    landslide[
        source_row - 1:source_row + 2,
        source_col - 1:source_col + 2
    ] = 1

    frontier = landslide.astype(bool).copy()
    frames = [landslide.copy()]

    for step in range(max_iterations):
        new_cells = np.zeros((rows, cols), dtype=bool)
        active_rows, active_cols = np.where(frontier)

        for r, c in zip(active_rows, active_cols):
            for dr, dc in neighbors:
                nr = r + dr
                nc = c + dc

                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                if landslide[nr, nc] == 1:
                    continue

                if susceptibility[nr, nc] < susceptibility_threshold:
                    continue

                drop = elevation[r, c] - elevation[nr, nc]

                if drop <= 0:
                    continue

                slope_factor = np.clip(drop / 3.0, 0.0, 1.0)

                probability = (
                    0.15
                    + 0.35 * susceptibility[nr, nc]
                    + 0.50 * slope_factor
                )

                if rng.random() < probability:
                    new_cells[nr, nc] = True

        if not np.any(new_cells):
            print(f"{name}: berhenti pada iterasi {step + 1}.")
            break

        landslide[new_cells] = 1
        frontier = new_cells
        frames.append(landslide.copy())

    else:
        print(f"{name}: mencapai batas iterasi maksimum.")

    results[name] = {
        "elevation": elevation,
        "frames": frames,
        "final": landslide.copy(),
        "iterations": len(frames) - 1,
        "cells": int(landslide.sum())
    }

for name, result in results.items():
    fig1, ax1 = plt.subplots(figsize=(5, 5))

    terrain_plot = ax1.imshow(
        result["elevation"],
        cmap="terrain"
    )

    ax1.set_title(f"Topografi - {name}")
    ax1.set_xlabel("Kolom")
    ax1.set_ylabel("Baris")

    fig1.colorbar(
        terrain_plot,
        ax=ax1,
        fraction=0.046,
        pad=0.04
    )

    fig1.tight_layout()
    plt.show(block=True)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5, 5))

    img = ax2.imshow(
        result["frames"][0],
        cmap="binary",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax2.set_title(f"{name} - Iterasi 0")
    ax2.set_xlabel("Kolom")
    ax2.set_ylabel("Baris")

    def animate(i, img=img, ax=ax2, result=result, name=name):
        img.set_data(result["frames"][i])
        ax.set_title(f"{name} - Iterasi {i}")
        return [img]

    anim = FuncAnimation(
        fig2,
        animate,
        frames=len(result["frames"]),
        interval=100,
        blit=False,
        repeat=False,
        cache_frame_data=False
    )

    fig2.tight_layout()
    plt.show(block=True)
    plt.close(fig2)

print("\nRingkasan hasil simulasi")

for name, result in results.items():
    print(
        f"{name}: "
        f"{result['cells']} sel terdampak, "
        f"{result['iterations']} iterasi"
    )