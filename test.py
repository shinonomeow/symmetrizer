import numpy as np
# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = np.array([[7, 8], [9, 10], [11, 12]])
# print(np.matmul(a, b))
#
# # a.shape is (2,3,4)
a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
print(a.shape)
print(a)
b = np.arange(2 * 3 * 4).reshape(2, 4, 3)
# b = np.arange(4 * 5).reshape(4, 5)
print(b.shape)
print(b)
c = np.matmul(a, b)
print(c)
print(c.shape)
# a 2x3x4x5
# c 
