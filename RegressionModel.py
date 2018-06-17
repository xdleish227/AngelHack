
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob
import os
import time

import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

from pprint import pprint


# In[4]:


#Load all training data(.csv) into dataframe
path = r'./train'
all_files = glob.glob(os.path.join(path, "*csv"))
df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True, sort=False)
print("Number of training data: " + str(df.shape))


# In[5]:


#feature engineering and data cleaning
df=df[[
    "ORIGIN_AIRPORT_ID",
    "DEST_AIRPORT_ID",
    "CRS_DEP_TIME",
    "MONTH",
    "DAY_OF_MONTH",
    "ARR_DEL15"
]].dropna(axis=0, how='any')

print("Feature engineering and data cleaning: " + str(df.shape))


# In[6]:


#Check if the assumption for logistic regression is correct
#sb.countplot(x="ARR_DEL15", data=df)


# In[9]:


#Split dataset in training and test datasets
XY_train, XY_test = train_test_split(df, test_size=0.1, random_state=int(time.time()))
#The test_size is 10%


# In[10]:


Y_train = XY_train[[
    "ARR_DEL15"
]].dropna(axis=0, how='any')

Y_test = XY_test[[
    "ARR_DEL15"
]].dropna(axis=0, how='any')

X_train = XY_train.drop(columns=["ARR_DEL15"])
X_test = XY_test.drop(columns=["ARR_DEL15"])


# In[11]:


print("Y_train size: " + str(Y_train.shape))
print("Y_test size: " + str(Y_test.shape))
print("X_train size: " + str(X_train.shape))
print("X_test size: " + str(X_test.shape))


# In[12]:


LogReg = LogisticRegression()
LogReg.fit(X_train, Y_train)
print(LogReg.score(X_train, Y_train))


# In[13]:


y_pred = LogReg.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(Y_test, y_pred))

y_prob = LogReg.predict_proba(X_test)
print(y_prob[:5])


# In[14]:


# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          Y_test.shape[0],
          (Y_test["ARR_DEL15"] != y_pred).sum(),
          100*(1-(Y_test["ARR_DEL15"] != y_pred).sum()/Y_test.shape[0])
))


# In[15]:


import pickle
LogRegression_Model_pkl = open("LogRegression_Model.pkl", "wb")
pickle.dump(LogReg, LogRegression_Model_pkl)
LogRegression_Model_pkl.close()

