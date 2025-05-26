import numpy as np
import matplotlib.pylab as plt

# 数值微分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def tangent_line(f, x):
    k = numerical_diff(f, x) # 斜率
    b = f(x) - k * x
    return lambda t: k * t + b



x = np.arange(0.0, 20.0, 0.1)
y1 = function_1(x)
tf = tangent_line(function_1, 10)
y2 = tf(x)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()

# 偏导数
def function_2(x):
    return np.sum(x ** 2)

# 计算梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_value = x[idx]
        x[idx] = tmp_value + h
        fx1 = f(x)
        x[idx] = tmp_value - h
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * h)
        x[idx] = tmp_value
    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))