import numpy as np

A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
print(np.ndim(A))
print(A.shape)

A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[1, 2, 3], [4, 5, 6]])
print(B.shape)
print(np.dot(A, B))

C = np.array([1, 2])
print(np.dot(A, C))

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
print(np.dot(X, W))
