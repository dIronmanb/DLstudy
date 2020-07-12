#matplotlib

import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0, 10, 0.1) #0에서 6까지 간격 0.1로 생성 -> 0, 0.1, 0.2 ...
y = np.sin(x) #x의 각각의 sin값이 나옴

plt.plot(x,y) #그래프 그리기
plt.show()    #그래프 보여주기
