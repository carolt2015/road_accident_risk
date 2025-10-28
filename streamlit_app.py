import streamlit as st

st.title('Road Accident Risk Prediction')

st.write('Choose variables to predict the risk:')

with st.sidebar:
  st.header('Input features')
  curvature  = st.slider('Road Curve',0,1)
