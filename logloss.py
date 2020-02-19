import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0.0000001, 1, 0.01)
y = -np.log(x)
plt.plot(x, y)
plt.xlabel('model output')
plt.ylabel('log loss')
plt.show()