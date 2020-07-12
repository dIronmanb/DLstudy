#mean squared error

import numpy as np

def mean_squared_error(y,t):
    return 0.5 * np.sum((y - t )**2)

#정답은 Index[2] (one-hot-encoding)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
p1 = mean_squared_error(np.array(y),np.array(t)) #0.0975
print(p1)


#만약 Index[7]에 있을 확률이 가장 높다고 했을 때
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.0, 0.1, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
p2 = mean_squared_error(np.array(y),np.array(t)) #0.6975
print(p2)




