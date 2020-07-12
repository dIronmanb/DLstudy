#Cross Entropy Error
import numpy as np

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+ delta))
    #log에 0이 나와서 -inf가 되는 것을 방지하기 위해서


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
p1 = cross_entropy_error(np.array(y),np.array(t))
print(p1) #0.510825457099338



y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
p2 = cross_entropy_error(np.array(y),np.array(t))
print(p2) #2.302584092994546
