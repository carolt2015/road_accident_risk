import streamlit as st

st.title('Road Accident Risk Prediction')

st.write('The risk prediction for your chosen values is:')

with st.sidebar:
  st.header('Choose variables for prediction')
  curvature  = st.slider('Road Curve',0,1)
