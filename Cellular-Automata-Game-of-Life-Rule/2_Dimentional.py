import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cellpylib as cpl
from custom_rule import rule

cellular_automaton = cpl.init_simple2d(200, 200)

# gliders
cellular_automaton[:, [28, 29, 30, 30], [30, 31, 29, 31] ]= 1

# blinkers
cellular_automaton[:, [40, 40, 40], [15, 16, 17] ]= 1

# light weights space ship (LWSS)
cellular_automaton[:, [18, 18, 19, 20, 21, 21, 21, 21, 20], [45, 48, 44, 44, 44, 45, 46, 47, 48] ]= 1



cellular_automaton = cpl.evolve2d(cellular_automaton, 
                                  timesteps=500, 
                                  memoize='recursive',
                                  apply_rule=rule,
                                  neighbourhood="Moore", 
                                  )

fig, ax = plt.subplots()
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
img = ax.imshow(cellular_automaton[0], cmap='binary', interpolation='nearest')

def init():
    img.set_data(cellular_automaton[0])
    return [img,]

def animate(i):
    img.set_data(cellular_automaton[i])
    return [img,]

animation = FuncAnimation(fig, animate, init_func=init, frames=500, interval=50, blit=True, repeat=True)

plt.show()
