import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

ROWS = 100
COLS = 100

ITERATIONS = 100

grid = np.zeros((ROWS, COLS), dtype=int)

start_row = ROWS // 2
start_col = COLS // 2

grid[start_row, start_col] = 1


def get_neighbors(row, col):

    neighbors = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:

            if dr == 0 and dc == 0:
                continue

            new_row = row + dr
            new_col = col + dc

            if 0 <= new_row < ROWS and 0 <= new_col < COLS:

                neighbors.append((new_row, new_col))

    return neighbors

def random_move(grid):

    active_cells = np.argwhere(grid == 1)

    new_grid = grid.copy()

    for row, col in active_cells:

        neighbors = get_neighbors(row, col)

        if neighbors:

            new_row, new_col = random.choice(neighbors)

            new_grid[new_row, new_col] = 1

    return new_grid


fig, ax = plt.subplots()

image = ax.imshow(grid, interpolation="nearest")

ax.set_title("Cellular Automata - Random Propagation")

ax.set_xlabel("Column")
ax.set_ylabel("Row")


def update(frame):

    global grid

    grid = random_move(grid)

    image.set_data(grid)

    ax.set_title(
        f"Cellular Automata - Iteration {frame + 1}"
    )

    return [image]


animation = FuncAnimation(
    fig,
    update,
    frames=ITERATIONS,
    interval=150,
    repeat=False
)


plt.show()