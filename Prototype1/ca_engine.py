
import numpy as np
import rasterio

from rasterio.windows import Window
from rasterio.windows import transform as window_transform


def run_ca(
    dem,
    transform,
    crs,
    source_center,
    source_size,
    critical_slope=15.0,
    exponent=1.0,
    window_radius=150,
    max_iterations=100,
    material_per_cell=1.0,
    epsilon=1e-12
):

    if dem is None:
        raise ValueError("DEM belum tersedia.")

    if source_center is None:
        raise ValueError("Source area belum dipilih.")

    if crs is None or not crs.is_projected:
        raise ValueError("DEM harus menggunakan CRS proyeksi.")

    if transform.a <= 0 or transform.e >= 0:
        raise ValueError("DEM harus memiliki orientasi north-up.")

    if source_size < 1 or source_size % 2 == 0:
        raise ValueError("Source size harus berupa bilangan ganjil.")

    if critical_slope <= 0:
        raise ValueError("Critical slope harus lebih besar dari 0.")

    if exponent <= 0:
        raise ValueError("MFD exponent harus lebih besar dari 0.")

    center_row, center_col = source_center

    half_size = source_size // 2

    row_start = center_row - half_size
    row_end = row_start + source_size

    col_start = center_col - half_size
    col_end = col_start + source_size

    if (
        row_start < 0
        or col_start < 0
        or row_end > dem.shape[0]
        or col_end > dem.shape[1]
    ):
        raise ValueError("Source area berada di luar DEM.")

    source_dem = dem[
        row_start:row_end,
        col_start:col_end
    ]

    if not np.all(np.isfinite(source_dem)):
        raise ValueError("Source area mengandung NoData.")

    window_row_start = max(
        0,
        center_row - window_radius
    )

    window_row_end = min(
        dem.shape[0],
        center_row + window_radius + 1
    )

    window_col_start = max(
        0,
        center_col - window_radius
    )

    window_col_end = min(
        dem.shape[1],
        center_col + window_radius + 1
    )

    local_dem = dem[
        window_row_start:window_row_end,
        window_col_start:window_col_end
    ].copy()

    local_window = Window(
        window_col_start,
        window_row_start,
        window_col_end - window_col_start,
        window_row_end - window_row_start
    )

    local_transform = window_transform(
        local_window,
        transform
    )

    dx = abs(local_transform.a)
    dy = abs(local_transform.e)

    if dx <= 0 or dy <= 0:
        raise ValueError("Resolusi DEM tidak valid.")

    local_source_row = (
        center_row - window_row_start
    )

    local_source_col = (
        center_col - window_col_start
    )

    local_row_start = (
        local_source_row - half_size
    )

    local_row_end = (
        local_row_start + source_size
    )

    local_col_start = (
        local_source_col - half_size
    )

    local_col_end = (
        local_col_start + source_size
    )

    local_source_dem = local_dem[
        local_row_start:local_row_end,
        local_col_start:local_col_end
    ]

    if not np.all(np.isfinite(local_source_dem)):
        raise ValueError("Source area tidak valid pada window simulasi.")

    rows, cols = local_dem.shape

    directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1)
    ]

    mobile_current = np.zeros(
        (rows, cols),
        dtype=np.float64
    )

    retained_total = np.zeros(
        (rows, cols),
        dtype=np.float64
    )

    source_mask = np.zeros(
        (rows, cols),
        dtype=bool
    )

    source_mask[
        local_row_start:local_row_end,
        local_col_start:local_col_end
    ] = True

    mobile_current[source_mask] = material_per_cell

    runout_footprint = source_mask.copy()

    initial_material = float(
        np.sum(mobile_current)
    )

    frames = [
        (
            mobile_current.copy(),
            runout_footprint.copy(),
            retained_total.copy()
        )
    ]

    iteration_results = []

    stop_reason = "Maximum iteration reached"

    for iteration in range(1, max_iterations + 1):

        mobile_next = np.zeros(
            (rows, cols),
            dtype=np.float64
        )

        active_cells = np.argwhere(
            mobile_current > epsilon
        )

        routed_cells = 0
        stopped_cells = 0
        routed_material = 0.0

        for row, col in active_cells:

            current_material = mobile_current[
                row,
                col
            ]

            current_elevation = local_dem[
                row,
                col
            ]

            eligible_neighbors = []
            weights = []

            for dr, dc in directions:

                nr = row + dr
                nc = col + dc

                if not (
                    0 <= nr < rows
                    and 0 <= nc < cols
                ):
                    continue

                neighbor_elevation = local_dem[
                    nr,
                    nc
                ]

                if not np.isfinite(neighbor_elevation):
                    continue

                elevation_difference = (
                    current_elevation - neighbor_elevation
                )

                if elevation_difference <= 0:
                    continue

                distance = np.sqrt(
                    (dc * dx) ** 2
                    + (dr * dy) ** 2
                )

                slope_ratio = (
                    elevation_difference / distance
                )

                slope_angle = np.degrees(
                    np.arctan(slope_ratio)
                )

                if slope_angle < critical_slope:
                    continue

                weight = slope_ratio ** exponent

                eligible_neighbors.append(
                    (nr, nc)
                )

                weights.append(weight)

            if len(eligible_neighbors) == 0:

                retained_total[row, col] += (
                    current_material
                )

                stopped_cells += 1

                continue

            weights = np.asarray(
                weights,
                dtype=np.float64
            )

            proportions = weights / np.sum(weights)

            distributed_material = 0.0

            for index, (nr, nc) in enumerate(
                eligible_neighbors
            ):

                if index == len(eligible_neighbors) - 1:

                    amount = (
                        current_material
                        - distributed_material
                    )

                else:

                    amount = (
                        current_material
                        * proportions[index]
                    )

                    distributed_material += amount

                mobile_next[nr, nc] += amount

                if amount > epsilon:
                    runout_footprint[nr, nc] = True

            routed_cells += 1
            routed_material += current_material

        total_mobile = float(
            np.sum(mobile_next)
        )

        total_retained = float(
            np.sum(retained_total)
        )

        total_material = (
            total_mobile + total_retained
        )

        conserved = np.isclose(
            total_material,
            initial_material,
            rtol=1e-8,
            atol=1e-8
        )

        if not conserved:
            raise RuntimeError(
                "Material conservation failed "
                f"at iteration {iteration}."
            )

        iteration_results.append({
            "iteration": iteration,
            "active": len(active_cells),
            "routed": routed_cells,
            "stopped": stopped_cells,
            "routed_material": routed_material,
            "mobile": total_mobile,
            "retained": total_retained,
            "conserved": conserved
        })

        mobile_current = mobile_next

        frames.append(
            (
                mobile_current.copy(),
                runout_footprint.copy(),
                retained_total.copy()
            )
        )

        if total_mobile <= epsilon:

            stop_reason = "No mobile material remains"
            break

        boundary_touched = (
            np.any(runout_footprint[0, :])
            or np.any(runout_footprint[-1, :])
            or np.any(runout_footprint[:, 0])
            or np.any(runout_footprint[:, -1])
        )

        if boundary_touched:

            stop_reason = (
                "Simulation window boundary reached. "
                "Consider increasing window radius."
            )

            break

    footprint_cells = int(
        np.count_nonzero(runout_footprint)
    )

    footprint_area = (
        footprint_cells * dx * dy
    )

    final_mobile = float(
        np.sum(mobile_current)
    )

    final_retained = float(
        np.sum(retained_total)
    )

    return {
        "dem": local_dem,
        "transform": local_transform,
        "dx": dx,
        "dy": dy,
        "source_mask": source_mask,
        "frames": frames,
        "iteration_results": iteration_results,
        "initial_material": initial_material,
        "final_mobile": final_mobile,
        "final_retained": final_retained,
        "footprint": runout_footprint,
        "footprint_cells": footprint_cells,
        "footprint_area": footprint_area,
        "stop_reason": stop_reason,
        "iterations": len(iteration_results)
    }
