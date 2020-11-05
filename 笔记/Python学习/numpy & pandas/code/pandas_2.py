import pandas as pd 
import numpy as np

dates = pd.date_range('20200101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index = dates, \
    columns=['A','B','C','D'])


df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan

# drop 0: 按行处理， 1：按列处理
print(df.dropna(axis=0, how='all')) # how={'any', 'all'}

# fill
print(df.fillna(value=0))

print(np.any(df.isnull()) == True)