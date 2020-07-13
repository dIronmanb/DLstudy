import numpy as np
import functions as f
import gradient as gd


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = gd.numerical_gradient(f,x)
        x -= lr * grad

    return x

def function(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
array = gradient_descent(function, init_x = init_x, lr=0.1, step_num=100)
print(array)

#학습률이 너무 크면 큰 값으로 발산한다.
#너무 작은 값은 거의 갱신되지 않은 태로 끝나버린다.
