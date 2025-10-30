import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Assuming pkl's are in the same directory as the app.py
model_path = Path(__file__).parent / "model.pkl"
scaler_path = Path(__file__).parent / "scaler.pkl"
poly_path = Path(__file__).parent / "poly.pkl"

# Cache and load the trained model
@st.cache_resource
def load_model():
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
    
# Cache and load the min_max scaler    
@st.cache_resource
def load_scaler():
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler 
    
# Cache and load the polynomial features used for training
@st.cache_resource
def load_poly():
    with open(poly_path, 'rb') as file:
        poly = pickle.load(file)
    return poly 

road_model = load_model() 
minmax_scaler = load_scaler()
poly_features = load_poly()

st.title('Road Accident Risk Prediction')

with st.sidebar:
  st.header('Choose conditions for safe driving')
  curvature  = st.slider('Road Curvature',0.0,1.0)
  speed_limit = st.slider('Speed Limit', 25,70)
  lighting = st.selectbox('Lighting',('daylight','dim', 'night'))
  weather = st.selectbox('Weather',('rainy','clear', 'foggy'))
  # Selectbox with True and False as options  
  road_signs_present = st.selectbox('Road Signs Present',options=[True, False],index=0)

  # For 'Predict' button  
  predicted = st.button("Predict")  
    

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
st.write(df_road[:1])
# Transform variable
df_road['speed_limit'] = np.log(df_road['speed_limit'])
st.write(df_road[:1])
scaler = MinMaxScaler(feature_range=(0,1))
#df_road['speed_limit'] = minmax_scaler.fit_transform(df_road[['speed_limit']])
df_scaled = df_road.copy()
df_scaled ['speed_limit'] = scaler.fit_transform(df_road['speed_limit'])

st.write(df_scaled[:1])

# Transform data with polynomial features
#df_road_poly =  poly_features.fit(df_road)
#poly_features.transform(df_road_poly)
#st.write("df_road_poly",df_road_poly.shape)


# Input data for prediction
input_row = df_road_poly[:1]
st.write(input_row.shape)
    
    
if predicted:
    prediction = road_model.predict(input_row)
    st.success(f"The chance of road accident is :{prediction[0] * 100}%")
    #if prediction[0] >= 0.5:
    #    st.subheader("Please drive safely!")
    #else:
    #    st.subheader("That's great!")



                                                          
                          
