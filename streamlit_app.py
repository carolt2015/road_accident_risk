import streamlit as st
import pickle

<!-- load the trained model -->

st.title('Road Accident Risk Prediction')
st.write('The chance of of road accident is :')

with st.sidebar:
  st.header('Choose variables for prediction')
  curvature  = st.slider('Road Curvature',0.0,1.0)
  speed_limit = st.slider('Speed Limit', 25,70)
  lighting = st.selectbox('Lighting',('daylight','dim', 'night'))
  weather = st.selectbox('Weather',('rainy','clear', 'foggy'))
  road_signs_present = st.selectbox('Road Signs Present',('True','False'))
                          
