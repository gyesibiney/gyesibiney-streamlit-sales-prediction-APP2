import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import joblib

# web page 
# title
st.title('Favorita Store Sales Prediction APP with Facebook Prophet')
st.markdown('this predict sales')

# data loading
model= joblib.load('saved_ml.joblib')


# inputs
st.header('make a forecast here:')
ds= st.date_input(label='Please enter your forecast date')
transactions= st.number_input(label='Please enter your total expected number of transactions')
onpromotion= st.number_input(label='Please enter total number of items on promotion')


# input dataframe
ok= st.button('forecast sales')
if ok:
    input_data= [ds,onpromotion,transactions]
    inputs= pd.DataFrame([input_data],columns=['ds','onpromotion','transactions'])
    # making Prediction
    forecast=model.predict(inputs)
    output_values=forecast['yhat']
    st.success (f'the estimated forecast sales ${output_values.values[0]:.2f}') 