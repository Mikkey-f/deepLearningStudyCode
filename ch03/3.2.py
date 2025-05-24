import numpy as np
import matplotlib.pyplot as plt
# 3.2.2 阶跃函数 只支持实数
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function2(x):
    y = x >= 0
    return y.astype(np.int_)

print(step_function2(np.array([-1.56, 0.0, 1.0])))

# 更简洁一点
def step_function3(x):
    return np.array(x > 0, dtype=np.int_)
x = np.arange(-5.0, 5.0, 0.1)
y = step_function3(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.5)
plt.show()

# sigmoid函数实现
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.5)
plt.show()

# ReLU函数
def relu(x):
    return np.maximum(0, x)
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.5)
plt.show()

