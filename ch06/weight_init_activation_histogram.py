import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100) # 生成1000个数据
node_num = 100
hidden_layer_size = 5
activation = {} # 激活值的结果保存到此

for i in range(hidden_layer_size):
    if i != 0:
        x = activation[i - 1]
    #w = np.random.randn(node_num, node_num) * 1
    # Xavier是激活函数为线性函数的前提下推导出来的
    w = np.random.rand(node_num, node_num) / np.sqrt(node_num)
    z = np.dot(x, w)
    a = np.tanh(z)
    activation[i] = a

# 绘制直方图
for i, a in activation.items():
    plt.subplot(1, len(activation), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()