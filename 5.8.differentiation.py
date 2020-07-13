### 컴퓨터는 해석적 미분을 하는 것이 안된다

import numpy as np

#미분-수치미분
"""
def numerical_diff(f,x): #Python - first citizen
    h = 1e-4    #0.0001로만 해줘도 충분
    return (f(x+h) - f(x-h) / (2*h)) #도함수 정의에 의하여 -중심 차분
"""

#편미분(gradient: 모든 변수의 편미분)
def numerical_diff(f,x): #인수 x에는 넘파이 배열이 들어감
    h = 1e-4
    grad = np.zeros_like(x) #x와 형상이 같은 배열 형성

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #값 복원

    return grad

##### 편미분 예제 #####
def function_2(x):
    return x[0]**2 + x[1]**2

r1 = numerical_diff(function_2, np.array([3.0, 4.0]))
r2 = numerical_diff(function_2, np.array([2.0, 0.0]))
r3 = numerical_diff(function_2, np.array([1.0, 5.0]))

print(r1, r2, r3, sep="\n")
                    
