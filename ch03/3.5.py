import numpy as np

def softmax(x):
    c = np.max(x)
    return np.exp(x - c) / np.sum(np.exp(x - c))

a = np.array([0.3, 2.9, 4.0])
print(softmax(a))
print(np.sum(softmax(a)))


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

a = np.array([3, 2, 4])
print(_change_one_hot_label(a))