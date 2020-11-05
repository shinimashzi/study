import pandas as pd 
import numpy as np

dates = pd.date_range('20200101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index = dates, \
    columns=['A','B','C','D'])
# print(df)


# 切片,选择行
print(df[0:3])

# select by label:loc
print(df.loc['20200102'])
print(df.loc['20200102':'20200105',:]) 
# df.loc[1:2,:] and df.loc[:,1:2] 会报错，只能使用它设定的索引

# select by position: iloc
print(df.iloc[:,1])
print(df.iloc[[1,3,5],1:3])

# mixed selection: ix, 1.0.0 已经弃用
# print(df.ix[:3,['A','C']])

# Boolean index
print(df[df.A > 8])
