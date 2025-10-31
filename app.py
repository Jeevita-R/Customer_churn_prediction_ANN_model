import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
st.header('Welcome to Customer Churn Prediction!!!')
df=pd.read_csv('cleaned.csv')
model=tf.keras.models.load_model('model.h5')
with open('labelencoder.pkl','rb') as file:
    le=pickle.load(file)
with open('onehotencoder.pkl','rb') as file:
    ohe=pickle.load(file)
with open('standardscaler.pkl','rb') as file:
    ss=pickle.load(file)
st.set_page_config(page_title='Customer churn prediction')
with st.container(border=True):
    CreditScore=st.number_input('Credit_score:')
    Geography=st.selectbox('Geography:',options=df['Geography'].unique())
    Gender=st.selectbox('Gender:',df['Gender'].unique())
    Age=st.slider('Age:',min_value=0,max_value=100)
    Tenure=st.number_input('Tenure:')
    Balance=st.number_input('Balance:')
    NumOfProducts=st.number_input('NumOfProducts:')
    HasCrCard=st.selectbox('HasCrCard:',options=[1,0])
    IsActiveMember=st.selectbox('IsActiveMember:',options=[1,0])
    EstimatedSalary=st.number_input('EstimatedSalary:')
    
    input_data={
    'CreditScore':CreditScore,
    'Geography':Geography,
    'Gender':Gender,
    'Age':Age,
    'Tenure':Tenure,
    'Balance':Balance,
    'NumOfProducts':NumOfProducts,
    'HasCrCard':HasCrCard,
    'IsActiveMember':IsActiveMember,
    'EstimatedSalary':EstimatedSalary     

    }
    data=ohe.transform([[input_data['Geography']]])
    in_df=pd.DataFrame(data,columns=ohe.get_feature_names_out(['Geography']))  
    in_df1=pd.DataFrame(input_data,index=[0])
    input_data=pd.concat([in_df1,in_df],axis=1)
    input_data=input_data.drop('Geography',axis=1)
    input_data['Gender']=le.transform(input_data['Gender'])
    in_data=ss.transform(input_data)
    prediction=model.predict(in_data)
    pred=prediction[0][0]
    if st.button('Submit'):
        if pred>=0.5:
            st.subheader('custmer likely to churn')
        else:
            st.subheader('not likely to churn')