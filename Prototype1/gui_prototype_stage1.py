
import sys
import os
import traceback
import numpy as np
import rasterio

from rasterio.transform import array_bounds

from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Rectangle, Patch
from matplotlib.animation import FuncAnimation

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QSlider
)

from PySide6.QtCore import Qt, QThread, Signal

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT
)

from matplotlib.figure import Figure

from ca_engine import run_ca


class SimulationWorker(QThread):

    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        dem,
        transform,
        crs,
        source_center,
        source_size,
        critical_slope,
        exponent
    ):

        super().__init__()

        self.dem = dem
        self.transform = transform
        self.crs = crs
        self.source_center = source_center
        self.source_size = source_size
        self.critical_slope = critical_slope
        self.exponent = exponent

    def run(self):

        try:

            result = run_ca(
                dem=self.dem,
                transform=self.transform,
                crs=self.crs,
                source_center=self.source_center,
                source_size=self.source_size,
                critical_slope=self.critical_slope,
                exponent=self.exponent,
                window_radius=150,
                max_iterations=100,
                material_per_cell=1.0
            )

            self.finished.emit(result)

        except Exception:

            self.error.emit(
                traceback.format_exc()
            )


class MapCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):

        self.figure = Figure(
            figsize=(9, 7),
            facecolor="#FFFFFF"
        )

        super().__init__(self.figure)

        self.setParent(parent)

        self.ax = None
        self.images = []
        self.animation = None

        self.dem = None
        self.transform = None
        self.crs = None
        self.extent = None
        self.hillshade = None

        self.source_patch = None
        self.source_center = None
        self.source_size = 5

        self.selection_mode = False
        self.simulation_result = None

        self._pan_active = False
        self._pan_start = None
        self._pan_limits = None

        self.current_frame = 0
        self.frame_count = 0

        self.on_coordinate_changed = None
        self.on_source_selected = None
        self.on_frame_changed = None

        self.setup_terrain_view()

        self.mpl_connect(
            "motion_notify_event",
            self.on_mouse_move
        )

        self.mpl_connect(
            "button_press_event",
            self.on_map_click
        )

        self.mpl_connect(
            "button_release_event",
            self.on_mouse_release
        )

        self.mpl_connect(
            "scroll_event",
            self.on_scroll
        )

    def setup_terrain_view(self):

        self.figure.clear()

        self.ax = self.figure.add_subplot(111)

        self.ax.set_facecolor("#E8E8E8")

        self.ax.text(
            0.5,
            0.5,
            "LOAD DEM TO START",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="#555555"
        )

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.images = []

        self.draw_idle()

    def stop_animation(self):

        if self.animation is not None:

            try:
                self.animation.event_source.stop()
            except Exception:
                pass

            self.animation = None

    def load_dem(self, file_path):

        self.stop_animation()

        with rasterio.open(file_path) as src:

            dem_masked = src.read(
                1,
                masked=True
            )

            self.dem = dem_masked.astype(
                np.float64
            ).filled(np.nan)

            self.transform = src.transform
            self.crs = src.crs

            west, south, east, north = array_bounds(
                src.height,
                src.width,
                src.transform
            )

            self.extent = [
                west,
                east,
                south,
                north
            ]

        valid_dem = np.isfinite(self.dem)

        if not np.any(valid_dem):
            raise ValueError(
                "DEM tidak memiliki data valid."
            )

        dem_for_shading = np.where(
            valid_dem,
            self.dem,
            np.nanmedian(self.dem)
        )

        light_source = LightSource(
            azdeg=315,
            altdeg=35
        )

        hillshade = light_source.hillshade(
            dem_for_shading,
            vert_exag=1.15,
            dx=abs(self.transform.a),
            dy=abs(self.transform.e)
        )

        valid_shade = hillshade[valid_dem]

        low = np.percentile(valid_shade, 1)
        high = np.percentile(valid_shade, 99)

        if high > low:

            hillshade = np.clip(
                (hillshade - low) / (high - low),
                0,
                1
            )

        hillshade = 0.5 + (
            hillshade - 0.5
        ) * 0.55

        hillshade = np.clip(
            hillshade + 0.12,
            0,
            1
        )

        hillshade[~valid_dem] = np.nan

        self.hillshade = hillshade

        self.source_center = None
        self.source_patch = None
        self.simulation_result = None

        self.show_terrain()

    def show_terrain(self, preserve_view=False):

        self.stop_animation()

        previous_xlim = None
        previous_ylim = None

        if (
            preserve_view
            and self.ax is not None
            and self.dem is not None
        ):
            previous_xlim = self.ax.get_xlim()
            previous_ylim = self.ax.get_ylim()

        self.figure.clear()

        self.ax = self.figure.add_subplot(111)

        self.ax.imshow(
            self.hillshade,
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=self.extent,
            origin="upper",
            interpolation="nearest"
        )

        self.ax.set_title(
            "DEMNAS - Shaded Relief",
            color="#000000"
        )

        self.ax.set_xlabel(
            "UTM Easting (m)",
            color="#000000"
        )

        self.ax.set_ylabel(
            "UTM Northing (m)",
            color="#000000"
        )

        self.ax.tick_params(
            colors="#000000",
            labelsize=8
        )

        self.ax.set_aspect("equal")

        self.ax.grid(
            True,
            alpha=0.15,
            color="#555555"
        )

        self.images = []
        self.source_patch = None

        if self.source_center is not None:

            self.draw_source(
                self.source_center[0],
                self.source_center[1]
            )

        if previous_xlim is not None and previous_ylim is not None:
            self.ax.set_xlim(previous_xlim)
            self.ax.set_ylim(previous_ylim)

        self.draw_idle()

    def set_selection_mode(self, enabled):

        self.selection_mode = enabled

    def set_source_size(self, size):

        self.source_size = size

        if self.source_center is not None:

            row, col = self.source_center

            half_size = size // 2

            row_start = row - half_size
            row_end = row_start + size

            col_start = col - half_size
            col_end = col_start + size

            if (
                row_start >= 0
                and col_start >= 0
                and row_end <= self.dem.shape[0]
                and col_end <= self.dem.shape[1]
                and np.all(
                    np.isfinite(
                        self.dem[
                            row_start:row_end,
                            col_start:col_end
                        ]
                    )
                )
            ):

                self.draw_source(row, col)

            else:

                self.source_center = None
                self.source_patch = None

                self.show_terrain(preserve_view=True)

                if self.on_source_selected:
                    self.on_source_selected(
                        None,
                        None,
                        None
                    )

    def on_map_click(self, event):

        if event.inaxes != self.ax:
            return

        if not self.selection_mode:
            if event.button == 1:
                self._pan_active = True
                self._pan_start = (event.x, event.y)
                self._pan_limits = (
                    self.ax.get_xlim(),
                    self.ax.get_ylim()
                )
            return

        if event.xdata is None or event.ydata is None:
            return

        if self.dem is None:
            return

        col_float, row_float = (
            ~self.transform
        ) * (
            event.xdata,
            event.ydata
        )

        row = int(np.floor(row_float))
        col = int(np.floor(col_float))

        if not (
            0 <= row < self.dem.shape[0]
            and 0 <= col < self.dem.shape[1]
        ):
            return

        half_size = self.source_size // 2

        row_start = row - half_size
        row_end = row_start + self.source_size

        col_start = col - half_size
        col_end = col_start + self.source_size

        if (
            row_start < 0
            or col_start < 0
            or row_end > self.dem.shape[0]
            or col_end > self.dem.shape[1]
        ):
            return

        source_dem = self.dem[
            row_start:row_end,
            col_start:col_end
        ]

        if not np.all(np.isfinite(source_dem)):
            return

        self.source_center = (row, col)

        self.show_terrain(preserve_view=True)

        self.draw_source(row, col)

        if self.on_source_selected:

            self.on_source_selected(
                row,
                col,
                float(self.dem[row, col])
            )

    def draw_source(self, row, col):

        if self.source_patch is not None:

            try:
                self.source_patch.remove()
            except ValueError:
                pass

        half_size = self.source_size // 2

        row_start = row - half_size
        col_start = col - half_size

        x = (
            self.transform.c
            + col_start * self.transform.a
        )

        y = (
            self.transform.f
            + (row_start + self.source_size)
            * self.transform.e
        )

        width = self.source_size * self.transform.a
        height = self.source_size * abs(self.transform.e)

        self.source_patch = Rectangle(
            (x, y),
            width,
            height,
            linewidth=2.2,
            edgecolor="#8B0000",
            facecolor="#FF2020",
            alpha=0.85,
            linestyle="-",
            zorder=20
        )

        self.ax.add_patch(self.source_patch)

        self.draw_idle()

    def on_mouse_release(self, event):

        if event.button == 1:
            self._pan_active = False
            self._pan_start = None
            self._pan_limits = None

    def on_scroll(self, event):

        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if event.button == "up":
            scale = 1 / 1.2
        elif event.button == "down":
            scale = 1.2
        else:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x = event.xdata
        y = event.ydata

        new_width = (xlim[1] - xlim[0]) * scale
        new_height = (ylim[1] - ylim[0]) * scale

        x_fraction = (x - xlim[0]) / (xlim[1] - xlim[0])
        y_fraction = (y - ylim[0]) / (ylim[1] - ylim[0])

        self.ax.set_xlim(
            x - new_width * x_fraction,
            x + new_width * (1 - x_fraction)
        )
        self.ax.set_ylim(
            y - new_height * y_fraction,
            y + new_height * (1 - y_fraction)
        )

        self.draw_idle()

    def on_mouse_move(self, event):

        if self._pan_active and self._pan_start is not None and self._pan_limits is not None:
            if event.x is not None and event.y is not None:
                start_x, start_y = self._pan_start
                start_xlim, start_ylim = self._pan_limits

                pixel_dx = event.x - start_x
                pixel_dy = event.y - start_y

                bbox = self.ax.get_window_extent()

                data_dx = -pixel_dx * (
                    (start_xlim[1] - start_xlim[0]) / bbox.width
                )
                data_dy = -pixel_dy * (
                    (start_ylim[1] - start_ylim[0]) / bbox.height
                )

                self.ax.set_xlim(
                    start_xlim[0] + data_dx,
                    start_xlim[1] + data_dx
                )
                self.ax.set_ylim(
                    start_ylim[0] + data_dy,
                    start_ylim[1] + data_dy
                )
                self.draw_idle()

        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if self.on_coordinate_changed:

            self.on_coordinate_changed(
                event.xdata,
                event.ydata
            )

    def create_overlay(self, mask, color, alpha):

        cmap = ListedColormap([color])

        cmap.set_bad(
            color=(0, 0, 0, 0)
        )

        masked = np.ma.masked_where(
            ~mask,
            mask.astype(float)
        )

        return self.ax.imshow(
            masked,
            cmap=cmap,
            vmin=0,
            vmax=1,
            alpha=alpha,
            extent=self.local_extent,
            origin="upper",
            interpolation="nearest"
        )

    def show_simulation(self, result):

        self.stop_animation()

        self.simulation_result = result

        self.figure.clear()

        self.ax = self.figure.add_subplot(111)

        dem = result["dem"]
        transform = result["transform"]

        rows, cols = dem.shape

        west, south, east, north = array_bounds(
            rows,
            cols,
            transform
        )

        self.local_extent = [
            west,
            east,
            south,
            north
        ]

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
            dx=result["dx"],
            dy=result["dy"]
        )

        valid_shade = hillshade[valid_dem]

        low = np.percentile(valid_shade, 1)
        high = np.percentile(valid_shade, 99)

        if high > low:

            hillshade = np.clip(
                (hillshade - low) / (high - low),
                0,
                1
            )

        hillshade = np.clip(
            0.5 + (hillshade - 0.5) * 0.55 + 0.12,
            0,
            1
        )

        hillshade[~valid_dem] = np.nan

        frames = result["frames"]

        self.frame_count = len(frames)
        self.current_frame = 0

        self.ax.imshow(
            hillshade,
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=self.local_extent,
            origin="upper",
            interpolation="nearest"
        )

        self.ax.set_xlim(west, east)
        self.ax.set_ylim(south, north)

        self.ax.set_aspect("equal")

        self.ax.set_xlabel(
            "UTM Easting (m)",
            color="#000000"
        )

        self.ax.set_ylabel(
            "UTM Northing (m)",
            color="#000000"
        )

        self.ax.tick_params(
            colors="#000000",
            labelsize=8
        )

        self.ax.grid(
            True,
            color="#555555",
            alpha=0.12
        )

        self.ax.set_facecolor("#E8E8E8")

        initial_mobile, initial_footprint, initial_retained = frames[0]

        epsilon = 1e-12

        footprint_image = self.create_overlay(
            initial_footprint,
            "#2E8B57",
            0.65
        )

        retained_image = self.create_overlay(
            initial_retained > epsilon,
            "#F2C94C",
            0.90
        )

        mobile_masked = np.ma.masked_where(
            initial_mobile <= epsilon,
            initial_mobile
        )

        mobile_cmap = plt_greys_cmap()

        mobile_image = self.ax.imshow(
            mobile_masked,
            cmap=mobile_cmap,
            vmin=0,
            vmax=max(
                float(np.max(frame[0]))
                for frame in frames
            ) or 1.0,
            alpha=0.95,
            extent=self.local_extent,
            origin="upper",
            interpolation="nearest"
        )

        source_mask = result["source_mask"]

        source_image = self.create_overlay(
            source_mask,
            "#FF2020",
            1.0
        )

        source_rows, source_cols = np.where(source_mask)

        if len(source_rows) > 0:

            rmin = int(np.min(source_rows))
            rmax = int(np.max(source_rows))
            cmin = int(np.min(source_cols))
            cmax = int(np.max(source_cols))

            source_x = (
                transform.c
                + cmin * transform.a
            )

            source_y = (
                transform.f
                + (rmax + 1) * transform.e
            )

            source_width = (
                (cmax - cmin + 1)
                * transform.a
            )

            source_height = (
                (rmax - rmin + 1)
                * abs(transform.e)
            )

            source_outline = Rectangle(
                (source_x, source_y),
                source_width,
                source_height,
                linewidth=2.5,
                edgecolor="#8B0000",
                facecolor="none",
                linestyle="-",
                zorder=20
            )
            self.ax.add_patch(source_outline)

        legend_items = [

            Patch(
                facecolor="#FF2020",
                edgecolor="#8B0000",
                label="Source Area"
            ),

            Patch(
                facecolor="#333333",
                edgecolor="#333333",
                label="Mobile Material"
            ),

            Patch(
                facecolor="#2E8B57",
                edgecolor="#2E8B57",
                label="Runout Footprint"
            ),

            Patch(
                facecolor="#F2C94C",
                edgecolor="#D6A800",
                label="Retained Material"
            )
        ]

        self.ax.legend(
            handles=legend_items,
            loc="upper right",
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="#CCCCCC",
            framealpha=0.92
        )

        self.images = [
            footprint_image,
            retained_image,
            mobile_image,
            source_image
        ]

        self.figure.suptitle(
            "Cellular Automata - Landslide Runout",
            fontsize=13,
            color="#000000",
            fontweight="bold"
        )

        self.figure.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        self.update_frame(0)

        self.animation = FuncAnimation(
            self.figure,
            self.update_frame,
            frames=self.frame_count,
            interval=300,
            repeat=False,
            blit=False,
            cache_frame_data=False
        )

        self.draw_idle()

    def update_frame(self, frame_index):

        if self.simulation_result is None:
            return self.images

        frame_index = int(
            np.clip(
                frame_index,
                0,
                self.frame_count - 1
            )
        )

        mobile, footprint, retained = (
            self.simulation_result["frames"][frame_index]
        )

        epsilon = 1e-12

        self.images[0].set_data(
            np.ma.masked_where(
                ~footprint,
                footprint.astype(float)
            )
        )

        self.images[1].set_data(
            np.ma.masked_where(
                retained <= epsilon,
                (retained > epsilon).astype(float)
            )
        )

        self.images[2].set_data(
            np.ma.masked_where(
                mobile <= epsilon,
                mobile
            )
        )

        self.current_frame = frame_index

        self.ax.set_title(
            f"Iteration {frame_index} / {self.frame_count - 1}",
            fontsize=11,
            color="#333333"
        )

        if self.on_frame_changed:

            self.on_frame_changed(
                frame_index
            )

        self.draw_idle()

        return self.images

    def set_frame_manually(self, frame_index):

        if self.animation is not None:

            if self.animation is not None and self.animation.event_source is not None:
                 self.animation.event_source.stop()

        self.update_frame(frame_index)


def plt_greys_cmap():

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("Greys").copy()

    cmap.set_bad(
        color=(0, 0, 0, 0)
    )

    return cmap


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Landslide Runout Simulator"
        )

        self.resize(1450, 900)

        self.dem_path = None
        self.worker = None
        self.simulation_result = None

        self.setup_ui()

    def setup_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(
            central_widget
        )

        main_layout.setContentsMargins(
            10, 10, 10, 8
        )

        main_layout.setSpacing(8)

        header = QHBoxLayout()

        title = QLabel(
            "LANDSLIDE RUNOUT SIMULATOR"
        )

        title.setStyleSheet(
            "font-size: 20px;"
            "font-weight: bold;"
            "color: #000000;"
            "background: transparent;"
        )

        subtitle = QLabel(
            "CA-based terrain simulation | Pujon, Malang"
        )

        subtitle.setStyleSheet(
            "color: #555555;"
            "font-size: 11px;"
            "background: transparent;"
        )

        header.addWidget(title)
        header.addStretch()
        header.addWidget(subtitle)

        main_layout.addLayout(header)

        content_layout = QHBoxLayout()

        content_layout.setSpacing(10)

        main_layout.addLayout(
            content_layout,
            stretch=1
        )

        left_panel = QWidget()

        left_panel.setFixedWidth(235)

        left_layout = QVBoxLayout(left_panel)

        left_layout.setContentsMargins(0, 0, 0, 0)

        data_group = QGroupBox("Data Acquisition")

        data_layout = QVBoxLayout(data_group)

        self.dem_label = QLabel("DEM: Not loaded")
        self.dem_label.setWordWrap(True)

        self.load_button = QPushButton("Load DEM")

        self.load_button.clicked.connect(
            self.load_dem
        )

        data_layout.addWidget(self.dem_label)
        data_layout.addWidget(self.load_button)

        source_group = QGroupBox("Source Selection")

        source_layout = QVBoxLayout(source_group)

        self.source_size = QSpinBox()

        self.source_size.setRange(1, 51)
        self.source_size.setSingleStep(2)
        self.source_size.setValue(5)

        self.source_size.valueChanged.connect(
            self.on_source_size_changed
        )

        self.select_button = QPushButton(
            "Select Source on Map"
        )

        self.select_button.setCheckable(True)

        self.select_button.toggled.connect(
            self.toggle_selection
        )

        self.source_info = QLabel(
            "Source: Not selected"
        )

        self.source_info.setWordWrap(True)

        source_layout.addWidget(
            QLabel("Source size (cells)")
        )

        source_layout.addWidget(self.source_size)
        source_layout.addWidget(self.select_button)
        source_layout.addWidget(self.source_info)

        parameter_group = QGroupBox("CA Parameters")

        parameter_layout = QVBoxLayout(parameter_group)

        self.critical_slope = QDoubleSpinBox()

        self.critical_slope.setRange(0.1, 89.9)
        self.critical_slope.setValue(15.0)
        self.critical_slope.setSuffix("°")

        self.mfd_exponent = QDoubleSpinBox()

        self.mfd_exponent.setRange(0.1, 10.0)
        self.mfd_exponent.setValue(1.0)
        self.mfd_exponent.setSingleStep(0.1)

        parameter_layout.addWidget(
            QLabel("Critical slope")
        )

        parameter_layout.addWidget(
            self.critical_slope
        )

        parameter_layout.addWidget(
            QLabel("MFD exponent")
        )

        parameter_layout.addWidget(
            self.mfd_exponent
        )

        self.run_button = QPushButton(
            "Run Simulation"
        )

        self.run_button.setEnabled(False)
        self.run_button.setMinimumHeight(38)

        self.run_button.clicked.connect(
            self.run_simulation
        )

        self.reset_button = QPushButton("Reset")

        self.reset_button.clicked.connect(
            self.reset_source
        )

        left_layout.addWidget(data_group)
        left_layout.addWidget(source_group)
        left_layout.addWidget(parameter_group)
        left_layout.addWidget(self.run_button)
        left_layout.addWidget(self.reset_button)
        left_layout.addStretch()

        map_panel = QWidget()

        map_layout = QVBoxLayout(map_panel)

        map_layout.setContentsMargins(0, 0, 0, 0)

        map_title = QLabel(
            "Terrain and Runout Visualization"
        )

        map_title.setStyleSheet(
            "font-weight: bold;"
            "font-size: 14px;"
            "color: #000000;"
            "background: transparent;"
        )

        self.canvas = MapCanvas()

        self.toolbar = NavigationToolbar2QT(
            self.canvas,
            self
        )

        map_layout.addWidget(map_title)
        map_layout.addWidget(self.toolbar)
        map_layout.addWidget(
            self.canvas,
            stretch=1
        )

        control_layout = QHBoxLayout()

        self.play_button = QPushButton("Pause")
        self.play_button.setEnabled(False)

        self.play_button.clicked.connect(
            self.toggle_play
        )

        self.replay_button = QPushButton("Replay")
        self.replay_button.setEnabled(False)

        self.replay_button.clicked.connect(
            self.replay_simulation
        )

        self.frame_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)

        self.frame_slider.valueChanged.connect(
            self.on_slider_changed
        )

        self.frame_value_label = QLabel(
            "0 / 0"
        )

        self.frame_value_label.setMinimumWidth(65)

        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.replay_button)
        control_layout.addWidget(self.frame_slider, stretch=1)
        control_layout.addWidget(self.frame_value_label)

        map_layout.addLayout(control_layout)

        right_panel = QWidget()

        right_panel.setFixedWidth(225)

        right_layout = QVBoxLayout(right_panel)

        right_layout.setContentsMargins(0, 0, 0, 0)

        result_group = QGroupBox("Simulation Results")

        result_layout = QVBoxLayout(result_group)

        self.iteration_label = QLabel("Iteration: -")
        self.mobile_label = QLabel("Mobile material: -")
        self.footprint_label = QLabel("Footprint area: -")
        self.retained_label = QLabel("Retained material: -")

        for label in [
            self.iteration_label,
            self.mobile_label,
            self.footprint_label,
            self.retained_label
        ]:

            label.setWordWrap(True)

            label.setStyleSheet(
                "padding: 5px;"
                "color: #000000;"
                "background: transparent;"
            )

            result_layout.addWidget(label)

        status_group = QGroupBox("Simulation Status")

        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel("Waiting for DEM")
        self.status_label.setWordWrap(True)

        status_layout.addWidget(self.status_label)

        right_layout.addWidget(result_group)
        right_layout.addWidget(status_group)
        right_layout.addStretch()

        content_layout.addWidget(left_panel)

        content_layout.addWidget(
            map_panel,
            stretch=1
        )

        content_layout.addWidget(right_panel)

        self.coordinate_label = QLabel(
            "Easting: - | Northing: -"
        )

        self.statusBar().addWidget(
            self.coordinate_label
        )

        self.statusBar().showMessage("Ready")

        self.canvas.on_coordinate_changed = (
            self.update_coordinates
        )

        self.canvas.on_source_selected = (
            self.update_source_info
        )

        self.canvas.on_frame_changed = (
            self.update_result_labels
        )

        self.apply_style()

    def apply_style(self):

        self.setStyleSheet("""
            QWidget {
                color: #000000;
                background-color: #F2F3F5;
                font-size: 10pt;
            }

            QMainWindow {
                background-color: #F2F3F5;
            }

            QLabel {
                color: #000000;
                background-color: transparent;
            }

            QGroupBox {
                color: #000000;
                font-weight: bold;
                border: 1px solid #D0D0D0;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                background-color: #FFFFFF;
            }

            QGroupBox::title {
                color: #000000;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                background-color: transparent;
            }

            QPushButton {
                color: #000000;
                padding: 7px;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                background-color: #FFFFFF;
            }

            QPushButton:hover {
                color: #000000;
                background-color: #E0E8F2;
            }

            QPushButton:checked {
                color: #000000;
                background-color: #D7E8FF;
                border: 1px solid #4682B4;
            }

            QPushButton:disabled {
                color: #777777;
                background-color: #E8E8E8;
                border: 1px solid #D0D0D0;
            }

            QSpinBox, QDoubleSpinBox {
                color: #000000;
                padding: 5px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                background-color: #FFFFFF;
            }

            QSlider::groove:horizontal {
                height: 5px;
                background: #D0D0D0;
                border-radius: 2px;
            }

            QSlider::handle:horizontal {
                background: #4682B4;
                border: 1px solid #356A96;
                width: 13px;
                margin: -5px 0;
                border-radius: 6px;
            }

            QStatusBar {
                color: #000000;
                background-color: #E5E5E5;
            }

            QToolBar {
                color: #000000;
                background-color: #EEEEEE;
                border: none;
                spacing: 4px;
            }

            QToolButton {
                color: #000000;
                background-color: #EEEEEE;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 3px;
            }

            QToolButton:hover {
                color: #000000;
                background-color: #DADADA;
                border: 1px solid #BBBBBB;
            }
        """)

    def load_dem(self):

        if self.worker is not None and self.worker.isRunning():

            QMessageBox.warning(
                self,
                "Simulation Running",
                "Tunggu sampai simulasi selesai."
            )

            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DEM Raster",
            "",
            "Raster Files (*.tif *.tiff)"
        )

        if not file_path:
            return

        try:

            self.canvas.load_dem(file_path)

            self.dem_path = file_path

            file_name = os.path.basename(file_path)

            self.dem_label.setText(
                "DEM: " + file_name
            )

            self.source_info.setText(
                "Source: Not selected"
            )

            self.run_button.setEnabled(False)

            self.reset_animation_controls()

            self.status_label.setText(
                "DEM loaded successfully. Select source area."
            )

            self.statusBar().showMessage(
                "DEM loaded successfully"
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "DEM Error",
                str(error)
            )

    def toggle_selection(self, enabled):

        if self.canvas.dem is None:

            self.select_button.setChecked(False)

            self.statusBar().showMessage(
                "Load DEM before selecting source."
            )

            return

        self.canvas.set_selection_mode(enabled)

        if enabled:

            self.select_button.setText(
                "Click Map to Select"
            )

            self.status_label.setText(
                "Click the map to select source center."
            )

        else:

            self.select_button.setText(
                "Select Source on Map"
            )

    def on_source_size_changed(self, size):

        self.canvas.set_source_size(size)

    def update_source_info(
        self,
        row,
        col,
        elevation
    ):

        if row is None:

            self.source_info.setText(
                "Source: Not selected"
            )

            self.run_button.setEnabled(False)

            return

        self.source_info.setText(
            f"Row: {row}\n"
            f"Column: {col}\n"
            f"Elevation: {elevation:.2f} m"
        )

        self.status_label.setText(
            "Source area selected successfully."
        )

        self.statusBar().showMessage(
            "Source area selected."
        )

        self.select_button.setChecked(False)

        self.run_button.setEnabled(True)

    def update_coordinates(self, x, y):

        self.coordinate_label.setText(
            f"Easting: {x:.2f} m | "
            f"Northing: {y:.2f} m"
        )

    def run_simulation(self):

        if self.canvas.dem is None:
            return

        if self.canvas.source_center is None:
            return

        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.source_size.setEnabled(False)

        self.play_button.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.frame_slider.setEnabled(False)

        self.status_label.setText(
            "Calculating CA simulation..."
        )

        self.statusBar().showMessage(
            "Running simulation..."
        )

        self.worker = SimulationWorker(
            dem=self.canvas.dem,
            transform=self.canvas.transform,
            crs=self.canvas.crs,
            source_center=self.canvas.source_center,
            source_size=self.canvas.source_size,
            critical_slope=self.critical_slope.value(),
            exponent=self.mfd_exponent.value()
        )

        self.worker.finished.connect(
            self.simulation_finished
        )

        self.worker.error.connect(
            self.simulation_error
        )

        self.worker.start()

    def simulation_finished(self, result):

        self.simulation_result = result

        self.canvas.show_simulation(result)

        last_frame = len(result["frames"]) - 1

        self.frame_slider.blockSignals(True)

        self.frame_slider.setRange(
            0,
            last_frame
        )

        self.frame_slider.setValue(0)

        self.frame_slider.blockSignals(False)

        self.frame_slider.setEnabled(
            last_frame > 0
        )

        self.play_button.setEnabled(
            last_frame > 0
        )

        self.replay_button.setEnabled(
            last_frame > 0
        )

        self.play_button.setText("Pause")

        self.status_label.setText(
            "Simulation completed.\n"
            + result["stop_reason"]
        )

        self.statusBar().showMessage(
            "Simulation completed."
        )

        self.load_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.source_size.setEnabled(True)
        self.run_button.setEnabled(True)

        self.worker = None

    def simulation_error(self, message):

        self.status_label.setText(
            "Simulation failed."
        )

        self.statusBar().showMessage(
            "Simulation error."
        )

        self.load_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.source_size.setEnabled(True)

        self.run_button.setEnabled(
            self.canvas.source_center is not None
        )

        self.worker = None

        QMessageBox.critical(
            self,
            "Simulation Error",
            message
        )

    def update_result_labels(self, frame_index):

        if self.simulation_result is None:
            return

        result = self.simulation_result

        mobile, footprint, retained = (
            result["frames"][frame_index]
        )

        mobile_total = float(
            np.sum(mobile)
        )

        retained_total = float(
            np.sum(retained)
        )

        footprint_area = (
            np.count_nonzero(footprint)
            * result["dx"]
            * result["dy"]
        )

        self.iteration_label.setText(
            f"Iteration: {frame_index}"
        )

        self.mobile_label.setText(
            f"Mobile material: {mobile_total:.4f}"
        )

        self.footprint_label.setText(
            f"Footprint area: {footprint_area:.2f} m²"
        )

        self.retained_label.setText(
            f"Retained material: {retained_total:.4f}"
        )

        self.frame_value_label.setText(
            f"{frame_index} / {len(result['frames']) - 1}"
        )

        self.frame_slider.blockSignals(True)

        self.frame_slider.setValue(
            frame_index
        )

        self.frame_slider.blockSignals(False)

    def on_slider_changed(self, value):

        if self.simulation_result is None:
            return

        self.canvas.set_frame_manually(value)

        self.play_button.setText("Play")

    def toggle_play(self):

        if self.canvas.animation is None:
            return

        if self.play_button.text() == "Pause":

            self.canvas.animation.event_source.stop()

            self.play_button.setText("Play")

        else:

            last_frame = self.canvas.frame_count - 1

            if self.canvas.current_frame >= last_frame:

                self.canvas.update_frame(0)

            self.canvas.animation.event_source.start()

            self.play_button.setText("Pause")

    def replay_simulation(self):

        if self.canvas.animation is None:
            return

        self.canvas.animation.event_source.stop()

        self.canvas.update_frame(0)

        self.canvas.animation.event_source.start()

        self.play_button.setText("Pause")

    def reset_animation_controls(self):

        if self.canvas.animation is not None:
            self.canvas.stop_animation()

        self.play_button.setEnabled(False)
        self.replay_button.setEnabled(False)

        self.play_button.setText("Pause")

        self.frame_slider.setEnabled(False)

        self.frame_slider.blockSignals(True)

        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)

        self.frame_slider.blockSignals(False)

        self.frame_value_label.setText("0 / 0")

    def reset_source(self):

        if self.worker is not None and self.worker.isRunning():
            return

        self.canvas.source_center = None
        self.canvas.source_patch = None

        self.canvas.simulation_result = None
        self.simulation_result = None

        self.canvas.selection_mode = False

        self.select_button.setChecked(False)

        self.select_button.setText(
            "Select Source on Map"
        )

        self.canvas.show_terrain()

        self.source_info.setText(
            "Source: Not selected"
        )

        self.iteration_label.setText("Iteration: -")
        self.mobile_label.setText("Mobile material: -")
        self.footprint_label.setText("Footprint area: -")
        self.retained_label.setText("Retained material: -")

        self.reset_animation_controls()

        self.run_button.setEnabled(False)

        self.status_label.setText(
            "Source selection reset."
        )

        self.statusBar().showMessage(
            "Source reset."
        )

    def closeEvent(self, event):

        if self.worker is not None and self.worker.isRunning():

            QMessageBox.warning(
                self,
                "Simulation Running",
                "Tunggu sampai simulasi selesai sebelum menutup aplikasi."
            )

            event.ignore()
            return

        event.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = MainWindow()

    window.show()

    sys.exit(app.exec())
