import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import tensorflow as tf
# from tensorflow.keras.models import load_model 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import streamlit as st


## Load the model
model = tf.keras.models.load_model('model.h5')


## Load all the pickle files

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file) 


##Create an streamlit app
st.title('Customer Churn Predictor')

# Example input data
# input_data = {
#     'CreditScore': 600,
#     'Geography': 'France',
#     'Gender': 'Male',
#     'Age': 40,
#     'Tenure': 3,
#     'Balance': 60000,
#     'NumOfProducts': 2,
#     'HasCrCard': 1,
#     'IsActiveMember': 1,
#     'EstimatedSalary': 50000
# }

# geography = st.selectbox('Geography:', onehot_encoder_geo.categories_[0])  # default first
# gender = st.selectbox('Gender:', label_encoder_gender.classes_)            # default first

# age = st.slider("Select Your Age:", min_value=18, max_value=80, value=30)
# tenure = st.slider("Tenure:", min_value=1, max_value=10, step=1, value=3)
# numproducts = st.slider("Number Of Products:", min_value=1, max_value=4, step=1, value=1)

# creditscore = st.number_input("Enter Your Credit Score:", min_value=0, value=650, step=1, format="%d")
# balance = st.number_input("Balance:", min_value=0, value=0, step=1, format="%d")
# estsalary = st.number_input("Estimated Salary:", min_value=0, value=50000, step=1, format="%d")


## User Inputs
geography = st.selectbox('Geography:', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender:', label_encoder_gender.classes_)

age = st.slider("Select Your Age:", min_value=18, max_value=80, value=25)
tenure = st.slider("Tenure:", min_value=1, max_value=10, step=1, value=3)
numproducts = st.slider("Number Of Products:", min_value=1, max_value=4, step=1, value=1)

creditscore = st.number_input("Enter Your Credit Score:", min_value=0, value=650, step=1, format="%d")
balance = st.number_input("Balance:", min_value=0, value=0, step=1, format="%d")
estsalary = st.number_input("Estimated Salary:", min_value=0, value=50000, step=1, format="%d")

has_cr_card = st.selectbox('Has Credit Card', options=['False', 'True'], index=1) ## To make 'True' the default selected option (which maps to 1)
is_active_member = st.selectbox('Is Active Member', options=['False', 'True'], index=1) ## To make 'True' the default selected option (which maps to 1)

# Mapping display values to integer values
# Convert to integer: 'False' -> 0, 'True' -> 1
has_cr_card_int = 1 if has_cr_card == 'True' else 0
is_active_member_int = 1 if is_active_member == 'True' else 0


##Compile the user inputs for further processing
# input_data = pd.DataFrame( {
#     'CreditScore': [creditscore],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [numproducts],
#     'HasCrCard': [has_cr_card_int],
#     'IsActiveMember': [is_active_member_int],
#     'EstimatedSalary': [estsalary]
#  } )
input_data = {
    'CreditScore': creditscore,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': numproducts,
    'HasCrCard': has_cr_card_int,
    'IsActiveMember': is_active_member_int,
    'EstimatedSalary': estsalary
 } 

##Convert the input_data to dataframe
input_df=pd.DataFrame([input_data])

##Label Encode Gender Categorical Variable
input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

## Scaling the input_data
input_scaled=scaler.transform(input_df)

## Predict the CHURN Now
prediction=model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')