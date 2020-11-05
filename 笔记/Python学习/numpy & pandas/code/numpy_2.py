import numpy as np

# A = np.array([1,1,1])
# B = np.array([2,2,2])

# # 上下合并
# C = np.vstack((A,B)) # np.vstack要求输入一个tuple
# print(C)
# # # 左右合并
# D = np.hstack((A,B))
# print(C.shape)
# print(D)

# print(A.T.shape)
# print(A[np.newaxis,:].shape) 
# print(A[:,np.newaxis])


# A = A[:,np.newaxis]
# B = B[:,np.newaxis]
# print(np.vstack((A,B)))
# C = np.concatenate((A,B,B), axis=1) # 多个array合并
# print(C)

# 均匀分割
A = np.arange(12).reshape((3,4))
print(A)
print(np.vsplit(A,3)) #v: vertical 分行
print(np.hsplit(A,2)) #h: horizontal 分列
# print(np.split(A,3,axis=0))

# 不等量分割
print(np.array_split(A,3,axis=1))
