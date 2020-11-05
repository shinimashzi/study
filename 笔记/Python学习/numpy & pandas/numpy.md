#  Python 学习之 Numpy

[TOC]

---

[视频教学内容](https://www.bilibili.com/video/BV1Ex411L7oT?p=10)

## 1. 基本内容 (numpy_0.py)

**基本属性**：

 ```python
 
# basic attributes of array
array = np.array([[1,2,3],
         [4,5,6]])
print(array.shape) # shape
print(array.ndim) # dimension
print(array.size) # size of the array
 ```

---

**创建array**：

```python
a = np.array([2,23,4], dtype=np.float)
a = np.array([[2,23,4],
              [2,32,4]])

a = np.zeros((3,4,5))
a = np.ones((3,4), dtype=np.int16)
a = np.empty((3,4))

a = np.arange(10,20,2) # [10 12 14 16 18]
a = np.arange(12).reshape((3,4))

a = np.linspace(1,10,6).reshape((2,3))
```

- 随机创建array

```python
# random create
a = np.arange(4).reshape((2,2))
a = np.array([[2,1],
             [0,3]])
print(a)
# np.sum()
# np.min()
# np.max()
print(np.max(a, axis=1)) # 对于2维数组，axis:1 -> 按行求和, axis = 0 -> 按列求和
```

- 关于axis：[参考这篇博客](https://www.cnblogs.com/cupleo/p/11330373.html)

> 设axis=i ,则numpy沿着第i个下标变化的方向进行操作。 from bilibili

**基础运算：**

- numpy-array的+-*/基本运算是对应每个位置进行的。

```python
# basic operation
a = np.array([10,20,30,40])
b = np.arange(4)
c = a-b
c = b**2 # [0 1 4 9]
c = 10*np.sin(a)
```

- array作为矩阵的运算：

```python
# matrix multiply
a = np.array([[1,1],
              [0,1]])
b = np.arange(4).reshape((2,2))
c = np.dot(a,b)
c = a.dot(b)
```

- numpy部分运算函数

```python
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
```



## 2. numpy索引 (numpy_1.py)

```python
A = np.arange(3,15).reshape((3,4))
print(A)
print(A[2])
print(A[2,1])
print(A[2,:])

for row in A:
    print(row)

for col in A.T:
    print(col)
print(A.flatten())
for item in A.flat: # A.flat return a iterator
    print(item)
```



## 3. numpy array合并&分割 (numpy_2.py)

**合并**

```python
import numpy as np

A = np.array([1,1,1])
B = np.array([2,2,2])

# 上下合并
C = np.vstack((A,B)) # np.vstack要求输入一个tuple
print(C)
# # 左右合并
D = np.hstack((A,B))
print(C.shape)
print(D)

print(A.T.shape)
print(A[np.newaxis,:].shape) 
print(A[:,np.newaxis])


A = A[:,np.newaxis]
B = B[:,np.newaxis]
print(np.vstack((A,B)))
C = np.concatenate((A,B,B), axis=0) # 多个array合并
print(C)
```

**分割**

```python
# 均匀分割
A = np.arange(12).reshape((3,4))
print(A)
print(np.vsplit(A,3)) 
print(np.hsplit(A,2)) 
# print(np.split(A,3,axis=0))

# 不等量分割
print(np.array_split(A,3,axis=1))
```



## 4. numpy copy & deep copy

numpy中的浅拷贝

```python
# copy
a = np.arange(4)
b = a
c = a
d = b
a[0] = 5

print(a)
print(b)
print(c)
print(d)

b[1] = 6

print(a)
print(b)
print(c)
print(d)

print(b is a) # True
print(d is a) # True
```

numpy中的深拷贝

```python

# deep copy
a = np.arange(4)
b = a.copy() # 复制值，
print(b is a) 
a[3] = 44
print(b)
```

