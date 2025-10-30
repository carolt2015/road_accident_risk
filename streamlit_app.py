import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
from pathlib import Path

# Assuming pkl is in the same directory as the app.py
model_path = Path(__file__).parent / "model.pkl"

# Cache and load the trained model
@st.cache_resource
def load_model():
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
    
road_model = load_model() 

st.title('Road Accident Risk Prediction')

with st.sidebar:
  st.header('Choose conditions for safe driving')
  curvature  = st.slider('Road Curvature',0.0,1.0)
  speed_limit = st.slider('Speed Limit', 25,70)
  lighting = st.selectbox('Lighting',('daylight','dim','night'))
  weather = st.selectbox('Weather',('rainy','clear','foggy'))
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
    
df = pd.DataFrame(data,index=[0])

# Adjust input format as per the model
# Encode variables
#encode = ['lighting','weather']
  
df_road = pd.get_dummies(df,columns=['lighting','weather']).astype(int)
#df_dummies[['lighting_daylight','lighting_dim','lighting_night','weather_rainy','weather_clear','weather_foggy']] = pd.DataFrame([[0],[0],[0],[0],[0],[0]],index=df.index)
#df_dummies = pd.DataFrame({'lighting_daylight':[0],'lighting_dim':[0],'lighting_night':[0],'weather_rainy':[0],'weather_clear':[0],'weather_foggy':[0]})
#df_road = pd.concat([df_road,df_dummies]) 
st.write("Shape of df_road after adding dummies",df_road.shape)
st.write(df_road[:1])

# Convert boolean variables to int
if road_signs_present == 'True':
    df_road['road_signs_present'] = 1
else:
    df_road['road_signs_present'] = 0

# Transform variable
df_road['speed_limit'] = np.log(df_road['speed_limit'])

scaler = MinMaxScaler(feature_range=(0,1))
df_road['speed_limit'] = scaler.fit_transform(df_road[['speed_limit']]).astype(float)


# Transform data with polynomial features
poly = PolynomialFeatures(degree=2)
st.write("df_road Shape",df_road.shape)

df_road_poly =  poly.fit_transform(df_road)

st.write("df_road_poly",df_road_poly.shape)

# Input data for prediction
input_row = df_road_poly[:1]
st.write("input_shape",input_row.shape)
    
    
if predicted:
    prediction = road_model.predict(input_row)
    st.success(f"The chance of road accident is :{prediction[0] * 100}%")
    #if prediction[0] >= 0.5:
    #    st.subheader("Please drive safely!")
    #else:
    #    st.subheader("That's great!")



                                                          
                          
