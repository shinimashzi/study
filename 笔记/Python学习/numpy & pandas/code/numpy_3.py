import numpy as np

copy
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

# 以上改变任意变量的值，会同时改变

# deep copy
a = np.arange(4)
b = a.copy() # 复制值，
print(b is a) 
a[3] = 44
print(b)