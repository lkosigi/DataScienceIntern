#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    ug = request.form.get('ug')
    ugyear = request.form.get('ugyear')
    pg = request.form.get('pg')
    pgyear = request.form.get('pgyear')
    python = request.form.get('python')
    r = request.form.get('r')
    ds = request.form.get('ds')
    ml = request.form.get('ml')
    dl = request.form.get('dl')
    nlp = request.form.get('nlp')
    sm = request.form.get('sm')
    aws = request.form.get('aws')
    sql = request.form.get('sql')
    nosql = request.form.get('nosql')
    xl= request.form.get('xl')
    
    if ug :
        type_UG= 1
    else:
        type_UG = 0
    if pg:
        type_PG= 1
        year = int(pgyear)
    else:
        type_PG = 0
        year = int(ugyear)
    if year == 2020:
        year_encoded = 3
    elif year == 2019:
        year_encoded = 2
    else :
        year_encoded = 1
    
    
    features = [int(python),int(r),int(ds),type_PG,type_UG,int(ml),int(dl),int(nlp),int(sm),int(aws),int(sql),int(nosql),int(xl),year_encoded]
    print(features)
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    if prediction[0] == 1:
        output = "Congrats! You are Shortlisted"
    elif prediction[0] == 0:
        output = "Sorry! You did not Qualify."
    
    return render_template('index.html', prediction_text='{}'.format(output))


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




