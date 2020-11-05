# Python 学习之 Pandas

[Toc]

`import pandas as pd`

## 1. 基础

- 创建及常见操作

```python

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
```

- 选择数据

```python
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
```

- 设置值

```python
dates = pd.date_range('20200101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index = dates, \
    columns=['A','B','C','D'])

df.iloc[2,2]=1111
print(df)
df[df.A>0]=0
print(df)
df['E'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20200101', periods=6)) # 设置index
df['F'] = np.nan
print(df)
```

- 处理丢失数据

```python
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
```

## 2. 导入、导出数据 & 合并 & 分割

```python
# read read_xxx
data = pd.read_csv('student.csv')
print(type(data))
print(data)

# save to_xxx
data.to_pickle('student.pickle')
```

## 3. plot

```python
# Series
data  = pd.Series(np.random.randn(1000), index = np.arange(1000))
data = data.cumsum()

# DataFrame
data = pd.DataFrame(np.random.randn(1000, 4),
                    index = np.arange(1000),
                    columns = list("ABCD"))
# plot methods
# bar, hist, box, kde, area, scatter,........
ax = data.plot.scatter(x='A',y='C',color='red',label='class')
bx = data.plot.scatter(x='A',y='B',color='DarkBlue', label='class1', ax=ax)
plt.show()
```

