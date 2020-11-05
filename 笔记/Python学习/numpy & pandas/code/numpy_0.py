import numpy as np

# basic attributes of array
array = np.array([[1,2,3],
         [4,5,6]])
# print(array.shape) # shape
# print(array.ndim) # dimension
# print(array.size) # size of the array

# create an array
a = np.array([2,23,4], dtype=np.float)
a = np.array([[2,23,4],
              [2,32,4]])

a = np.zeros((3,4,5))
a = np.ones((3,4), dtype=np.int16)
a = np.empty((3,4))

a = np.arange(10,20,2) # [10 12 14 16 18]
a = np.arange(12).reshape((3,4))

a = np.linspace(1,10,6).reshape((2,3))

# basic operation
a = np.array([10,20,30,40])
b = np.arange(4)
c = a-b
c = b**2 # [0 1 4 9]
c = 10*np.sin(a)

print(b<3) #b<3 return a list [True  True  True False]

# matrix operation
a = np.array([[1,1],
              [0,1]])
b = np.arange(4).reshape((2,2))
# matrix multiply
c = np.dot(a,b)
c = a.dot(b)

# random create
a = np.arange(4).reshape((2,2))
a = np.array([[2,1],
             [0,3]])
# np.sum()
# np.min()
# np.max()
print(np.max(a, axis=1)) # 对于2维数组，axis:1 -> 按行求和, axis = 0 -> 按列求和

A = np.arange(2,14).reshape((3,4))
print(A)
print(np.argmin(A)) # the index of the minimum
print(np.argmax(A)) 
print(np.mean(A))
print(np.average(A))
print(np.median(A))
print(np.cumsum(A)) # cumsum[i] = cumsum[i-1]+a[i]
print(np.diff(A)) # 相邻两个数之间的差
print(np.nonzero(A)) # 第一个array：行，第二个array：列
print(np.sort(A)) # 逐行排序
print(A.T) # 转置
print(np.clip(A,5,9)) # 所有小于5的数都变成5，大于9的数变成9

print(A.mean(axis=1))

