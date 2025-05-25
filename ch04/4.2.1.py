import numpy as np
# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))

# 交叉熵误差, 越小越准确
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum((t * np.log(y + delta)))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))

# mini-batch学习
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

def cross_entropy_error_one_hot_batch(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = np.reshape(y, y.size)
        t = np.reshape(t, t.size)
    batch_size = y[0]
    # 采用one_hot数据结构返回
    return -np.sum(t * np.log(y + delta)) / batch_size

def cross_entropy_error_index_batch(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = np.reshape(1, y.size)
        t = np.reshape(1, t.size)
    batch_size = y[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(x_batch.shape)
print(t_batch.shape)

# 在进行神经网络的学习时，不能将识别精度作为指标。因为如果以识别精度为指标，则参数的导数在绝大多数地方都会变为0。
# 寻找一个导数值非0的点对于神经网络至关重要，只有找到一个导数连续变化的曲线，才能对参数变化方向给出确定