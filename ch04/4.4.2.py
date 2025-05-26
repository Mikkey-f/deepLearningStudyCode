import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        # W指权重
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, y):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, z)
        return loss


net = simpleNet()
print(net.W)
# x为输入1*2
x = np.array([0.6, 0.9])
# p为隐藏神经元的输出1*3
p = net.predict(x)
print(p)
# 得到p输出的最大值
print(np.argmax(p))
t = np.array([0, 0, 1])
# 求解损失函数
print(net.loss(x, t))

# 求梯度，寻找到使得损失函数最小的权重
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)