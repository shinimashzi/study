import pandas as pd
import numpy as np


# create
s = pd.Series([1,3,6,np.nan,44,1])

dates = pd.date_range('20200101',periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4), index = dates, columns=['a','b','c','d']) # row : index, columns
print(df)

df = pd.DataFrame(np.arange(12).reshape((3,4)))
print(df)

df2 = pd.DataFrame({'A':1,
                    'B':pd.Timestamp('20130102'),
                    'C':np.array([3]*4,dtype='int32')},index=np.arange(4))

print(df2)
print(df2.dtypes)
print(df2.index) # Int64Index([0, 1, 2, 3], dtype='int64')
print(df2.columns) # Index(['A', 'B', 'C'], dtype='object')
print(df2.describe())
print(df2.sort_index(axis=0, ascending=False))
print(df2.sort_values(by='A'))