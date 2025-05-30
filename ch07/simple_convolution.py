import numpy as np

from common.util import im2col


class Convolution:
    def __init__(self, w, b, stride = 1, pad = 0):
        # w 为滤波器传入数据（数据量*通道*长*宽）
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad
    def forward(self, x):
        # x为传入数据
        fn, c, fh, fw = self.w.shape
        n, c, h, w = x.shape
        out_h = int(1 + (h + 2 * self.pad - fh) / self.stride)
        out_w = int(1 + (w + 2 * self.pad - fw) / self.stride)

        col = im2col(x, fh, fw, stride=self.stride, pad=self.pad)
        col_w = self.w.reshape(fn, -1).T
        out = np.dot(col, col_w) + self.b

        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out