import math

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, math.pi * 2, 0.05)

y = np.sin(x)
plt.plot(x, y, 'b')
plt.plot(x, np.cos(x), 'r')

plt.show()
