import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cellpylib as cpl


# ============================================================
# GRID
# ============================================================

ROWS = 100
COLS = 100

cellular_automaton = cpl.init_simple2d(ROWS, COLS)


# ============================================================
# FUNGSI MEMASANG PATTERN
# ============================================================

def place_pattern(grid, pattern, row, col, rotation=0, flip=False):

    pattern = np.array(pattern)

    # Flip horizontal
    if flip:
        pattern = np.fliplr(pattern)

    # Rotasi
    for _ in range(rotation):
        pattern = np.rot90(pattern)

    h, w = pattern.shape

    grid[:, row:row+h, col:col+w] = pattern


# ============================================================
# PATTERN
# ============================================================

GLIDER = [
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
]


BLINKER = [
    [1, 1, 1]
]


LWSS = [
    [0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1]
]


# ============================================================
# GLIDERS DARI BERBAGAI ARAH
# ============================================================

place_pattern(cellular_automaton, GLIDER, 5, 5)

place_pattern(cellular_automaton, GLIDER, 5, 90, flip=True)

place_pattern(cellular_automaton, GLIDER, 90, 5, rotation=2)

place_pattern(cellular_automaton, GLIDER, 90, 90, rotation=2, flip=True)


# ============================================================
# GLIDERS DI BAGIAN TENGAH
# ============================================================

place_pattern(cellular_automaton, GLIDER, 20, 45)

place_pattern(cellular_automaton, GLIDER, 45, 20, rotation=1)

place_pattern(cellular_automaton, GLIDER, 45, 75, rotation=3)

place_pattern(cellular_automaton, GLIDER, 75, 45, rotation=2)


# ============================================================
# LWSS
# ============================================================

place_pattern(cellular_automaton, LWSS, 10, 30)

place_pattern(
    cellular_automaton,
    LWSS,
    80,
    60,
    rotation=2
)

place_pattern(
    cellular_automaton,
    LWSS,
    40,
    5,
    rotation=1
)

place_pattern(
    cellular_automaton,
    LWSS,
    50,
    90,
    rotation=3
)


# ============================================================
# BLINKER - BAGIAN TENGAH
# ============================================================

place_pattern(
    cellular_automaton,
    BLINKER,
    48,
    48
)

place_pattern(
    cellular_automaton,
    BLINKER,
    52,
    48,
    rotation=1
)

place_pattern(
    cellular_automaton,
    BLINKER,
    48,
    52,
    rotation=1
)

place_pattern(
    cellular_automaton,
    BLINKER,
    52,
    52
)


# ============================================================
# RANDOM CLUSTERS
# ============================================================

np.random.seed(42)

for _ in range(15):

    row = np.random.randint(5, 90)
    col = np.random.randint(5, 90)

    random_pattern = np.random.randint(
        0,
        2,
        size=(5, 5)
    )

    place_pattern(
        cellular_automaton,
        random_pattern,
        row,
        col
    )


# ============================================================
# EVOLUTION
# ============================================================

cellular_automaton = cpl.evolve2d(
    cellular_automaton,
    timesteps=600,
    memoize='recursive',
    apply_rule=cpl.game_of_life_rule,
    neighbourhood="Moore"
)


# ============================================================
# VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(0, COLS)
ax.set_ylim(0, ROWS)

ax.set_xlabel("Columns")
ax.set_ylabel("Rows")

img = ax.imshow(
    cellular_automaton[0],
    cmap='binary',
    interpolation='nearest'
)


# ============================================================
# ANIMATION
# ============================================================

def init():

    img.set_data(cellular_automaton[0])

    ax.set_title(
        "Conway's Game of Life"
    )

    return [img]


def animate(i):

    img.set_data(cellular_automaton[i])

    ax.set_title(
        f"Conway's Game of Life - Generation {i}"
    )

    return [img]


animation = FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=600,
    interval=20,
    blit=True,
    repeat=False
)


plt.show()