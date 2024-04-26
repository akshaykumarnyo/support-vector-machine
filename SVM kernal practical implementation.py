#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-5.0,5.0,100)
y=np.sqrt(10**2-x**2)
y=np.hstack([y,-y])
x=np.hstack([x,-x])


# In[11]:


x1=np.linspace(-5.0,5.0,100)
y1=np.sqrt(5**2-x1**2)
y1=np.hstack([y1,-y1])
x1=np.hstack([x1,-x1])


# In[12]:


plt.scatter(y,x)
plt.scatter(y1,x1)


# In[17]:


import pandas as pd
df1=pd.DataFrame(np.vstack([y,x]).T,columns=["X1","X2"])
df1["Y"]=0
df2=pd.DataFrame(np.vstack([y1,x1]).T,columns=["X1","X2"])
df2["Y"]=1
df=pd.concat([df1,df2],ignore_index=True)
df.head(5)


# In[18]:


df.tail()


# In[20]:


## Polynomial  KErnal
## k(x,y)==(x^ty+c)**d


# In[21]:


## Based on the formula find the components for the Polynomial Kernal


# In[22]:


df["X1_Square"]=df["X1"]**2
df["X2_Square"]=df["X2"]**2
df["X1*X2"]=df["X1"]*df["X2"]
df.head()


# In[23]:


### independent anf Dependent 
x=df[["X1_Square","X2_Square","X1*X2"]]
y=df["Y"]


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[27]:


x_train.head()


# In[28]:


x_test


# In[29]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


get_ipython().system('pip install plotly')


# In[32]:


df.head()


# In[34]:


import plotly.express as px
import plotly.express as px
fig=px.scatter_3d(df,x="X1_Square",y="X2_Square",z="X1*X2",
                 color="Y")
fig.show()
plt.show()


# In[38]:


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
classifier=SVC(kernel="linear")
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
accuracy_score(y_test,y_pred)


# In[ ]:





# In[40]:


## Radial Basis Function Kernal


# In[ ]:


## 


# In[41]:


df.head()


# In[43]:


## Independent Feature
x=df.iloc[:,0:2]
## Dependent feature
y=df.Y


# In[44]:


y


# In[45]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[46]:


classifier=SVC(kernel="rbf")
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
accuracy_score(y_test,y_pred)


# In[51]:


classifier=SVC(kernel="poly")
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
accuracy_score(y_test,y_pred)


# In[ ]:





# In[52]:


## Sigmoid Kernel
### S(x)=1/1+e^-x==(e^x)/(e^x)+1


# In[53]:


classifier=SVC(kernel="sigmoid")
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




