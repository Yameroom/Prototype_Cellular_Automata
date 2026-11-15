import numpy as np
import matplotlib.pyplot as plt
import cellpylib as cpl
import matplotlib.animation as animation

class CustomRule(cpl.BaseRule):
    def __call__(self, n, c, t):
        neighbors = np.sum(n) - n[1][1]

        if n[1][1] == 0:
            if neighbors == 3:
                return 1
            return 0

        else:
            if 1 <= neighbors <= 5:
                return 1
            return 0

rule = CustomRule()
