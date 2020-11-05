import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# plot data

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