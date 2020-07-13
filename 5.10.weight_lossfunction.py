#가중치 매개변수에 관한 손실 함수의 기울기 (가중치의 형상과 같다)
import sys, os
sys.path.append(os.pardir)
import numpy as np
from functions import softmax, cross_entropy_error
from gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) #가중치 랜덤(2x3)

    def predict(self, x):
        return np.dot(x, self.W) #입력값 dot product 가중치

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z) #softmax함수로 출력값 구함
        loss = cross_entropy_error(y, t)
                        #정답 테이블과 출력값의 교차엔트로피오차 구함
        return loss


net = simpleNet()
print(net.W) #무작위로 선정된 가중치 값 출력

x = np.array([0.6, 0.9]) #x값 설정
t = np.array([0, 0, 1])  #정답 테이블 설정 (one_hot_encoding, 형상은 출력값과 같다)

p = net.predict(x) #입력값과 가중치의 내적
print(p)
print(np.argmax(p))

f = lambda w: net.loss(x, t) #손실함수 구함
dW = numerical_gradient(f, net.W) #손실함수를 가중치에 수치 미분

print(dW)
