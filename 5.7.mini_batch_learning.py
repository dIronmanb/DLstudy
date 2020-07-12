#Mini-batch(미니배치 학습) : 모집단에서 임의의 수 n개만큼 추출한다.
#그 n개의 이미지만 학습하는 것(대략적인 값을 기대할 수 있다.)

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist #이 dataset에서 추출

(x_train, t_train),(x_test,t_test) = \
          load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) #(60000, 784) - 훈련 데이터(60000개,28x28차원)
print(t_train.shape) #(60000, 10)  - 정답 레이블(60000개,10차원 데이터)

#무작위로 10장만 빼내기

train_size = x_train.shape[0] #[0]: 60000, [1]: 784
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) #60000개 중 10개 추출
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(batch_mask) #10개의 사진
print(x_batch)    #10개의 사진 각각의 input
print(t_batch)    #정답 label(ex) '[7]=1'에서는 숫자 7를 의미)

#(배치용) 교차 엔트로피 오차 구현하기

def cross_entropy_error(y, t):
    if y.ndim == 1:                 #y가 1차원이라면,
        t = t.reshape(1, t.size)    #형상 바꾸기
        y = y.reshape(1, y.size)    #형상 바꾸기

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y)) / batch_size #정규화

### 중요! 손실함수(비용함수)는 신경망을 올바르게 학습할 수 있도록 해준다 ###

