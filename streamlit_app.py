import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Assuming model.pkl is in the same directory as the app.py
model_path = Path(__file__).parent / "model.pkl"

# Cache and load the trained model
@st.cache_resource
def load_model():
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

road_model = load_model()  

st.title('Road Accident Risk Prediction')
st.write('The chance of of road accident is :')

with st.sidebar:
  st.header('Choose variables for prediction')
  curvature  = st.slider('Road Curvature',0.0,1.0)
  speed_limit = st.slider('Speed Limit', 25,70)
  lighting = st.selectbox('Lighting',('daylight','dim', 'night'))
  weather = st.selectbox('Weather',('rainy','clear', 'foggy'))
  # Selectbox with True and False as options  
  road_signs_present = st.selectbox('Road Signs Present',options=[True, False],index=0)
    

data  = { 'curvature':curvature, 
         'speed_limit':speed_limit,
         'lighting':lighting,
         'weather': weather, 
         'road_signs_present':road_signs_present
        }

input_data = pd.DataFrame(data, index=[0])

# Adjust input format as per the model
# Encode variables
encode = ['lighting','weather']
df_road = pd.get_dummies(input_data,columns=['lighting','weather'], prefix=encode,dtype=float)

# Convert boolean variables to int
df_road['road_signs_present'] = df_road['road_signs_present'].astype(int)

# Transform variable
df_road['speed_limit'] = np.log(df_road['speed_limit'])
scaler = MinMaxScaler(feature_range=(0,1))
df_road['speed_limit'] = scaler.fit_transform(df_road[['speed_limit']])


# Transform data with polynomial features
poly = PolynomialFeatures(2)
df_road_poly =  poly.fit_transform(df_road)
st.write("df_road",df_road.shape)
st.write("df_road_poly",df_road_poly.shape)

# Input data for prediction
input_row = df_road_poly[:1]

# For Debugging, remove after testing
st.write("input_shape",input_row.shape)
        
#prediction = road_model.predict(input_row)
st.subheader(f"The chance of of road accident is :{prediction[0]}")
                                                          
                          
