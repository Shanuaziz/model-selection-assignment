#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('titanic_dataset.csv')


# In[3]:


data


# In[4]:


data.columns


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[10]:


data.fillna(method='ffill', inplace=True)


# In[12]:


data


# In[13]:


data['PassengerId'].nunique()


# In[14]:


data=data.drop('PassengerId', axis=1)


# In[15]:


data


# In[16]:


data= pd.get_dummies(data)


# In[17]:


data


# In[30]:


X=data.drop('Survived',axis=1)
y= data['Survived']


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


LR_model=LogisticRegression()


# In[43]:


LR_model=LR_model.fit(X_train,y_train)


# In[44]:


score_LR=LR_model.score(X_test,y_test)


# In[45]:


score_LR


# In[46]:


from sklearn.model_selection import KFold
KFold_validator= KFold(10)
for train_index,test_index  in KFold_validator.split(X,y):
    print('Training Index: ',train_index)
    print('Testing Index: ', test_index)


# In[47]:


from sklearn.model_selection import cross_val_score
cv_result= cross_val_score(LR_model,X,y,cv=KFold_validator)


# In[48]:


cv_result


# In[49]:


from sklearn.model_selection import StratifiedKFold


# In[50]:


skfold_validator= StratifiedKFold(n_splits=10)


# In[51]:


for train_index,test_index  in skfold_validator.split(X,y):
    print('Training Index: ',train_index)
    print('Testing Index: ', test_index)


# In[52]:


cv_result= cross_val_score(LR_model,X,y,cv=skfold_validator)
cv_result


# In[53]:


np.mean(cv_result)


# In[54]:


from sklearn.svm import SVC


# In[55]:


svc_classifier=SVC()
svc_classifier.fit(X_train,y_train)


# In[59]:


X=data.drop('Survived',axis=1)
y= data['Survived']


# In[60]:


classifier = SVC()


# In[63]:


classifier.fit(X,y)


# In[64]:


svc_predictions= svc_classifier.predict(X_test)


# In[65]:


from sklearn.metrics import accuracy_score


# In[66]:


svc_accuracy=accuracy_score(y_test, svc_predictions)


# In[76]:


print("Support Vector Classifier Accuracy:", svc_accuracy)


# In[ ]:




