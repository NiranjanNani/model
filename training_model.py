#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier

df =pd.read_csv('train.csv')

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return "Unknown" 
    
def replace_titles(x):
    title=x["Title"]
    if title in ['Capt','Col','Major']:
        return 'Officer'
    elif title in ['Jonkheer','Don','the Countess','Dona','Lady','Sir']:
        return 'Royalty'
    elif title in ['the Countess','Mme','Lady']:
        return 'Mrs'
    elif title in ['Mlle','Ms']:
        return 'Miss'
    else:
        return title  
    
df["Title"]=df["Name"].map(lambda x:get_title(x))
df["Title"]=df.apply(replace_titles,axis=1)
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Fare'].fillna(df['Fare'].median(),inplace=True)
df['Embarked'].fillna("S",inplace=True)
df.drop("Cabin",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Name",axis=1,inplace=True)
df.Sex.replace(('male','female'),(0,1),inplace=True)
df.Embarked.replace(('S','C','Q'),(0,1,2),inplace=True)
df.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Royalty','Officer'),(0,1,2,3,4,5,6,7),inplace=True)

y=df["Survived"]
x=df.drop(['Survived','PassengerId'],axis=1)
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.1)
randomforest=RandomForestClassifier()
randomforest.fit(x_train,y_train)

filename='titanic_model_duplicate.sav'
pickle.dump(randomforest,open(filename,'wb'))


# In[ ]:




