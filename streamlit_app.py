import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from pathlib import Path

# Assuming pkl's is in the same directory as the app.py
model_path = Path(__file__).parent / "model.pkl"
dummies_path = Path(__file__).parent / "dummies.pkl"

# Cache and load trained model and dummies
@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        
    with open(dummies_path, 'rb') as file:
        dummies = pickle.load(file)    
    return model,dummies
    
road_model,road_dummies = load_resources() 

st.title('Road Accident Risk Prediction')

# User input 
with st.sidebar:
  st.header('Choose conditions for safe driving')
  curvature  = st.slider('Road Curvature',0.0,1.0)
  speed_limit = st.slider('Speed Limit', 25,70)
  lighting = st.selectbox('Lighting',['daylight','dim','night'])
  weather = st.selectbox('Weather',['rainy','clear','foggy'])
  # Selectbox with True and False as options  
  road_signs_present = st.selectbox('Road Signs Present',options=[True, False],index=0)

  # For 'Predict' button  
  predicted = st.button("Predict",type="primary")  
    
# User input as a dictionary
data  = { 'curvature':curvature, 
         'speed_limit':speed_limit,
         'lighting':lighting,
         'weather': weather, 
         'road_signs_present':road_signs_present
        }

# Convert user input to a dataframe    
df = pd.DataFrame(data,index=[0])

# Adjust input format as per the model
# Encode variables
df_dummies = pd.get_dummies(df,columns=['lighting','weather'])
# Reindex dataFrame to match the pickled dummies
df_road = df_dummies.reindex(columns=road_dummies, fill_value=0)

# Convert boolean variable to int
df_road['road_signs_present'] =  df_road['road_signs_present'] .astype(int)

# Transform and scale variable
df_road['speed_limit'] = np.log(df_road['speed_limit'])
scaler = MinMaxScaler(feature_range=(0,1))
df_road['speed_limit'] = scaler.fit_transform(df_road[['speed_limit']]).astype(float)

# Feature engineering data with polynomial features
poly = PolynomialFeatures(degree=2)
df_road_poly =  poly.fit_transform(df_road)

# Isolate input data for prediction
input_row = df_road_poly[:1]

# On click 'Predict' button
if predicted:
    prediction = road_model.predict(input_row)
    st.success(f"The chance of road accident is : {prediction[0] * 100:.2f}%")
   



                                                          
                          
