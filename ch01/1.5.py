# NumPy
import numpy as np

# 1维数组
x = np.array([1,2,3,4,5])
print(x)
print(type(x))
y = np.array([2,3,4,6,10])
print(x + y)
print(x - y)
print(x * y)
print(x / y)

# 矩阵
A = np.array([[1,2], [3,4]])
print(A)
# 矩阵形状
print(A.shape)
# 矩阵类型
print(A.dtype)
B = np.array([[1,2], [3,4]])
print(A + B)
print(A - B)
print(A * B)
# 广播
B = np.array([10, 20])
print(A * B)
## 访问元素
X = np.array([[1,2], [3,4]])
print(X[0])
print(X[0][1])
for i in X:
    print(i)
# 将X变为一维数组
X = X.flatten()
print(X)
print(X[np.array([0, 2])])
print(X[X != 1])