#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


## Lets create synthetic data point
from sklearn.datasets import make_regression


# In[5]:


x,y=make_regression(n_samples=1000,n_features=2,n_targets=1,noise=3.0)


# In[6]:


x


# In[7]:


y


# In[8]:


pd.DataFrame(x)[0]


# In[9]:


sns.scatterplot(x=pd.DataFrame(x)[0],y=pd.DataFrame(x)[1],hue=y)


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)


# In[13]:


from sklearn.svm import SVR


# In[14]:


svr=SVR(kernel="linear")


# In[15]:


svr.fit(x_train,y_train)


# In[16]:


svr.coef_


# In[17]:


## prediction
y_pred=svr.predict(x_test)


# In[19]:


y_pred


# In[21]:


from sklearn.metrics import r2_score


# In[22]:


print(r2_score(y_test,y_pred))


# In[23]:


### Hyperparameters Tuning With SVC


# In[32]:


from sklearn.model_selection  import GridSearchCV

## defining parameter range
param_grid={"C":[0.1,1,10,100,1000],
            "gamma":[1,0.1,0.01,0.001,0.0001],
            "kernel":["linear"],
            "epsilon":[0.1,0.2,0.3,0.4,0.5]
}


# In[33]:


grid=GridSearchCV(SVR(),param_grid=param_grid,refit=True,cv=5,verbose=3)


# In[34]:


grid.fit(x_train,y_train)


# In[35]:


grid.best_params_


# In[31]:


### Prediction


# In[38]:


y_pred=grid.predict(x_test)
print(r2_score(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




