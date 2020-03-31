#!/usr/bin/env python
# coding: utf-8

# In[2]:


import plotly


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')
X = [590,540,740,130,810,300,320,230,470,620,770,250] 
Y = [32,36,39,52,61,72,77,75,68,57,48,48]
plt.scatter(X,Y) 
plt.xlim(0,1000) 
plt.ylim(0,100)
#scatter plot color
plt.scatter(X, Y, s=60, c='red', marker='^')
#change axes ranges plt.xlim(0,1000) plt.ylim(0,100)
#add title
plt.title('Relationship Between Temperature and Iced Coffee Sales')
#add x and y labels
plt.xlabel('Sold Coffee') 
plt.ylabel('Temperature in Fahrenheit')
#show plot 
plt.show()


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np 
plt.style.use('seaborn-whitegrid')
# Create empty figure
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
# set the x and y axis range plt.xlim(0, 11)
plt.ylim(-2, 2) 
plt.axis('tight')
#add title
plt.title('Plotting data using sin and cos')


# In[14]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns 
plt.style.use('classic')
plt.style.use('seaborn-whitegrid') 
# Create some data
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])
# Plot the data with seaborn
sns.distplot(data['x']) 
sns.distplot(data['y']);


# In[15]:


for col in 'xy':
    sns.kdeplot(data[col], shade=True)


# In[16]:


sns.kdeplot(data);


# In[19]:


with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde');


# In[20]:


with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')


# In[21]:


sns.pairplot(data);


# In[90]:


import matplotlib.pyplot as plt 
import plotly.graph_objs as go 
import numpy as np
x = np.random.randn(2000)
y = np.random.randn(2000)

plt([go.Histogram2dContour(x=x, 
                        y=y, 
contours=dict (coloring='heatmap')), 
go.Scatter(x=x, 
           y=y, 
           mode='markers', marker=dict(color='white', size=3))],
            show_link=False)


# In[5]:


import plotly.offline as offline 
import plotly.graph_objs as go
offline.plot({'data': [{'y': [14, 22, 30, 44]}],
'layout': {'title': 'Offline Plotly', 'font':
dict(size=16)}}, image='png')


# In[83]:


import matplotlib.pyplot as plt
import plotly.graph_objs as go 
plt([go.Scatter(x=[95, 77, 84], y=[75, 67, 56])])


# In[37]:


import pandas as pd 
import numpy as np
df = pd.DataFrame(np.random.randn(200,6),index= pd.date_range
('1/9/2009', periods=200), columns= list('ABCDEF')) 
df.plot(figsize=(20, 10)).legend(bbox_to_anchor=(1, 1))


# In[6]:


import pandas as pd
df = pd.DataFrame(np.random.rand(20,5), columns=['Jan','Feb','March','April', 'May']) 
df.plot.bar(stacked=True, 
figsize=(20, 10)).legend(bbox_to_anchor=(1.1, 1))


# In[9]:


import pandas as pd
df = pd.DataFrame(np.random.rand(20,5), columns=['Jan','Feb', 'March','April', 'May']) 
df.plot.barh(stacked=True, figsize=(20, 10)).legend(bbox_to_anchor=(1.1, 1))


# In[15]:


import pandas as pd
df = pd.DataFrame(np.random.rand(20,5), columns=['Jan','Feb', 'March','April','May'])
df.plot.hist(bins= 20, figsize=(10,8)).legend 
bbox_to_anchor=(1.2,1)


# In[18]:


import pandas as pd
import numpy as np
df=pd.DataFrame({'April':np.random.randn(1000)+1,'May':np.random.
randn(1000),'June': np.random.randn(1000) - 1}, columns=['April', 'May', 'June'])
df.hist(bins=20)


# In[19]:


import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.rand(20,5),
columns=['Jan','Feb','March','April','May'])
df.plot.box()


# In[24]:


import pandas as pd
df = pd.DataFrame(np.random.rand(20,5),
columns= ['Jan','Feb','March','April', 'May'])
df.plot.area(figsize=(6, 4)).legend
bbox_to_anchor = (1.3, 1)


# In[25]:


import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.rand(20,5),columns= ['Jan','Feb', 'March','April', 'May'])
df.plot.scatter(x='Feb', y='Jan', title='Temperature over two months ')


# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
salesMen = ['Ahmed', 'Omar', 'Ali', 'Ziad', 'Salwa', 'Lila'] 
Mobile_Sales = [2540, 1370, 1320, 2000, 2100, 2150]
TV_Sales = [2200, 1900, 2150, 1850, 1770, 2000]
df = pd.DataFrame()
df ['Name'] =salesMen
df ['Mobile_Sales'] = Mobile_Sales 
df['TV_Sales']=TV_Sales
df.set_index("Name",drop=True,inplace=True)


# In[69]:


df.plot.bar( figsize=(20, 10), rot=0).legend(bbox_to_anchor=(1.1, 1)) 
plt.xlabel('Salesmen') 
plt.ylabel('Sales') 
plt.title('Sales Volume for two salesmen in \nJanuary and April 2017') 
plt.show()


# In[70]:


df.plot.pie(subplots=True)


# In[71]:


df.plot.box()


# In[72]:


df.plot.area(figsize=(6, 4)).legend(bbox_to_anchor=(1.3, 1))


# In[73]:


df.plot.bar(stacked=True, figsize=(20, 10)).legend (bbox_to_anchor=(1.1, 1))


# In[ ]:




