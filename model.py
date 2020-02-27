#!/usr/bin/env python
# coding: utf-8

# In[343]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# In[344]:


df = pd.read_excel("Data_Science_2020_v2.xlsx")


# In[ ]:


#df.head()
#df.count()
#df.isnull().sum()


# ## Handle missing values

# In[345]:


#handle missing values for other skills
df["Other skills"] = df['Other skills'].fillna('')


# In[346]:


# for Degree
df["Degree"].value_counts()


# In[347]:


# we will with empty cells with the majority
# from the value_counts we see that B.Tech is the most common degree
# hence use that value
df['Degree'] = df['Degree'].fillna("Bachelor of Technology (B.Tech)")


# In[348]:


#df["Python (out of 3)"].value_counts()


# ## Feature Extraction/Engineering

# In[349]:


# convert the rating into score/weights


# In[350]:


def rating_to_weighs(rating):
    if rating == 3:
        return 10
    elif rating == 2:
        return 7
    elif rating == 1:
        return 3
    else:
        return 0


# In[351]:


df["Weight"] = df["Python (out of 3)"].map(rating_to_weighs)


# In[352]:


df["Weight"] += df["R Programming (out of 3)"].map(rating_to_weighs)


# In[353]:


df["Weight"] += df["Data Science (out of 3)"].map(rating_to_weighs)


# In[354]:


df.head()


# In[355]:


# extract skills from column and add to the weight column


# In[356]:


def otherskill_to_score(skills):
    score=0
    if "Machine Learning" in skills:
        score +=3
    if "Deep Learning" in skills:
        score +=3
    if "NLP" in skills:
        score +=3
    if "Statistical Modeling" in skills :
        score +=3
    if "AWS" in skills:
        score +=3
    if "SQL" in skills:
        score +=3
    if "NoSQL" in skills:
        score +=3
    if "Excel" in skills:
        score +=3
    return score


# In[357]:


df['Weight'] += df["Other skills"].map(otherskill_to_score)


# In[358]:


# group the Degree into UG and PG for easy analysis
# from the data we see 6 catergories


# In[359]:


def degree_to_score(degree):
    type=""
    if "Bachelor" in degree:
        type="UG"
    elif "Master" in degree:
        type="PG"
    elif "MBA" in degree:
        type="PG"
    elif "B.Tech" in degree:
        type="UG"
    elif "Post Graduate" in degree or "PG" in degree:
        type="PG"
    elif "Integrated" in degree or "PG" in degree:
        type="PG"
    else:
        type="UG"
    return type
        


# In[360]:


df["type_of_degree"] = df['Degree'].map(degree_to_score)


# In[361]:


df['type_of_degree'].value_counts()


# In[362]:


# now add weights based on the type of degree and year of graduation


# In[363]:


def degree_type_year_to_score(degree_type, year):
    score = 0
    if degree_type == "UG" and year == 2020:
        score = 10
    elif degree_type == "UG" and year == 2019:
        score = 8
    elif degree_type == "UG" and year <= 2018:
        score = 5
    elif degree_type == "PG" and year == 2020:
        score = 7 
    elif degree_type == "PG" and year <= 2019:
        score = 3
    else:
        score = 0
    return score


# In[364]:


df['Weight'] += df.apply(lambda x: degree_type_year_to_score(x['type_of_degree'], x['Current Year Of Graduation']), axis=1)


# In[365]:


df[df["Weight"]>=40].count()


# In[366]:


#Create labels now, Greater than or equal to 40 are selected, and rest are not qualified


# In[367]:


def map_selected(weight):
    if weight >= 40:
        return 1
    else:
        return 0


# In[368]:


# After labelling data according tothe weights next,
#Preparing the data for modelling
# Here convert skills in text form to numerics to feed to the model, either using get_dummies() or map


# In[369]:


df = pd.get_dummies(data=df, columns=['type_of_degree'])


# In[370]:


def ml_to_dummy(skills):
    if "Machine Learning" in skills:
        return 1
    else :
        return 0
    


# In[371]:


df["ML"]= df['Other skills'].map(ml_to_dummy)


# In[372]:


def dl_to_dummy(skills):
    if "Deep Learning" in skills:
        return 1
    else :
        return 0


# In[373]:


df["DL"]= df['Other skills'].map(dl_to_dummy)


# In[374]:


def nlp_to_dummy(skills):
    if "NLP" in skills:
        return 1
    else :
        return 0


# In[375]:


df["NLP"]= df['Other skills'].map(nlp_to_dummy)


# In[376]:


def sm_to_dummy(skills):
    if "Statistical modeling" in skills:
        return 1
    else :
        return 0


# In[377]:


df["SM"]= df['Other skills'].map(sm_to_dummy)


# In[378]:


def aws_to_dummy(skills):
    if "AWS" in skills:
        return 1
    else :
        return 0


# In[379]:


df["AWS"]= df['Other skills'].map(aws_to_dummy)


# In[380]:


def sql_to_dummy(skills):
    if "SQL" in skills:
        return 1
    else :
        return 0
df["SQL"]= df['Other skills'].map(sql_to_dummy)


# In[381]:


def nosql_to_dummy(skills):
    if "NoSQL" in skills:
        return 1
    else :
        return 0
df["NoSQL"]= df['Other skills'].map(nosql_to_dummy)


# In[382]:


def xl_to_dummy(skills):
    if "Excel" in skills:
        return 1
    else :
        return 0
df["Excel"]= df['Other skills'].map(xl_to_dummy)


# In[383]:


# deal with year, with 2020 getting highest weight, followed by 2019, and the rest in a particular weights


# In[384]:


def year_to_dummy(year):
    if year == 2020:
        return 3
    elif year == 2019:
        return 2
    else:
        return 1


# In[385]:


df["year"]= df['Current Year Of Graduation'].map(year_to_dummy)


# In[386]:


df.info()


# In[387]:


df.columns


# In[388]:


# Create X,y matrix for modelling, retain only required numeric fields, drop the rest


# In[ ]:


X=df.drop(['Application_ID','Current City','Other skills', 'Institute', 'Degree', 'Stream','Current Year Of Graduation','Performance_PG', 'Performance_UG', 'Performance_12', 'Performance_10','Selected', 'Weight']  ,axis=1)
y=df['Selected']


# In[391]:


X.columns


# In[393]:


y.head()


# ## Logistic Regression Model for classification

# In[394]:


import sklearn.model_selection as model_selection
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=200)


# In[395]:


import sklearn.linear_model as linear_model
clf=linear_model.LogisticRegression()


# In[396]:


clf.fit(X,y)


# In[397]:


predictions = clf.predict(X_test)


# In[398]:


from sklearn.metrics import accuracy_score


# In[399]:


accuracy_score(y_test, predictions)


# ## Create a pickle file

# In[400]:


# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,0,3,0,1,1,1,1,0,0,0,0,0,2]]))


# In[ ]:




