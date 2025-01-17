import pandas as pd

import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle

model=tf.keras.models.load_model('model.h5')

with open('one_hot_encoder.pkl','rb') as file:
    one_hot_encoder=pickle.load(file)

with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)

with open('Standardscaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title("Customer Churn prediction")

geography=st.selectbox('Geography',one_hot_encoder.categories_[0])
gender=st.selectbox("Gender",label_encoder.classes_)
age=st.slider('Age',18,99)
balance=st.number_input('Bank balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('estimated_salary')
tenure=st.slider('Tenure',0,10)
no_of_products=st.slider('No of products',0,10)
has_cr_card=st.selectbox('Has credit card',[0,1])
is_active_member=st.selectbox('Is acive member',[0,1])

input_data=pd.DataFrame({
    
    "CreditScore": [credit_score],
    "Gender":[label_encoder.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[no_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_salary],
    })

geo=one_hot_encoder.transform([[geography]]).toarray()

OHE_geo=pd.DataFrame(geo,columns=one_hot_encoder.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),OHE_geo],axis=1)


scaled_data=scaler.transform(input_data)


# Predict churn
prediction = model.predict(scaled_data)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
