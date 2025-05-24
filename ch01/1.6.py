# matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

# x在 0-10 取值范围内的图形
# x = np.arange(0, 10, 0.1)
# # 自变量
# y = np.tan(x)
#
# plt.plot(x, y)
# plt.show()

# 绘制图形
x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
# x y1 曲线标签
plt.plot(x, y1, label='sin')
plt.plot(x, y2, label='cos')
plt.xlabel('x') # x轴标签
plt.ylabel('y')
plt.title('sin & cos')
# 右上角
plt.legend()
plt.show()

# 显示图像
img = imread('E:\\note\picture\\3.jpeg')
plt.imshow(img)
plt.show()
