#역전파(backward propagation)

"""
참고로, 지금까지 구현했던 신경망은 순전파(forward propagation)였다.
역전파를 통해 '미분'을 효율적으로 계산할 수 있다.
역전파로 흘러갈 때는 국소적 미분(편미분)을 곱하고 다음 노드로 전달한다.
"""

class MulLayer: #곱 역전파 (z = xy)
    def __init__(self): #생성자 - 멤버변수 선언
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        #역전파: x,y와 같이 뒤바뀐다.
        dx = dout*self.y    #미분 dx
        dy = dout*self.x    #미분 dy

        return dx, dy

class AddLayer: #합 연산자 (z = x+y)
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y

        return out

    def backward(self, dout):
        #역전파: 각각 편미분을 하면 1이 나온다.
        dx = dout * 1
        dy = dout * 1
        return dx, dy

    










        
