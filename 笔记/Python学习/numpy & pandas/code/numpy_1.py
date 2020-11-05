import numpy as np

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
