import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x)) #broadcast

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid_function(x)
plt.plot(x, y1)
plt.plot(x, y2)
plt.ylim(-0.1, 1.1) #y축의 범위 지정
plt.show()
