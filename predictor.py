#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("dataset/covid.csv")


# In[3]:


cormat = data.corr()
top_cor_features = cormat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_cor_features].corr(),annot=True,cmap="PRGn")


# In[4]:


A= ['Body_Temp', 'Breath_Rate', 'BloodPressure','Oxygen_Level','Vaccinated','Cough_Cold','Age']
B= ['Outcome']


# In[5]:


X = data[A].values
y = data[B].values


# In[6]:


X_pred = [[101,18,85,91,1,0,24]]
X_pred = pd.DataFrame(X_pred, columns=['Body_Temp', 'Breath_Rate', 'BloodPressure','Oxygen_Level','Vaccinated','Cough_Cold','Age'])


# In[7]:


model=RandomForestClassifier(n_estimators=100, n_jobs=-1)
model.fit(X,y)

prediction = model.predict(X_pred)
acc = metrics.accuracy_score(prediction,y[0])


# In[8]:


savename = "covidmodel.sav"
pickle.dump(model, open(savename, "wb"))


# In[9]:


load_model = pickle.load(open(savename, "rb"))
single = load_model.predict(X_pred)[0]
probability = load_model.predict_proba(X_pred)[:,1][0]*100
if single==1:
        output = "The patient is having Covid 19 Positive symptoms. Make sure to Quarantine Yourself from other and Stay Safe ."
        output1 = "Model Accuracy: {}".format(probability)
else:
    output = "The patient is having Covid 19 Negative symptoms. Maintain Social Distancing and Stay Safe."
    output1 = ""

print(output)
print(output1)

