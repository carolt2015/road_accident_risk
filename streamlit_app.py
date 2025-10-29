import streamlit as st
import pickle 
from sklearn.preprocessing import MinMaxScaler

# load the trained model 
try:
  with open('model.pkl', 'rb') as file:
    road_model = pickle.load(file) 
except FileNotFoundError:
            st.error("Model file 'model.pkl' not found. Please ensure it's in the correct path.")  

st.title('Road Accident Risk Prediction')
st.write('The chance of of road accident is :')

with st.sidebar:
  st.header('Choose variables for prediction')
  curvature  = st.slider('Road Curvature',0.0,1.0)
  speed_limit = st.slider('Speed Limit', 25,70)
  lighting = st.selectbox('Lighting',('daylight','dim', 'night'))
  weather = st.selectbox('Weather',('rainy','clear', 'foggy'))
  road_signs_present = st.selectbox('Road Signs Present',('True','False'))

data  = { 'curvature':curvature, 
         'speed_limit':speed_limit,
         'lighting':lighting,
         'weather': weather, 
         'road_signs_present':road_signs_present
        }

input_data = pd.Dataframe(data, index=[0])

# Adjust input format as per the model
# Encode variables
encode = ['lighting','weather']
df_road = pd.getdummies(input_data,prefix=encode)

# Convert boolean variables to int
df_road['road_signs_present'] = df_road['road_signs_present'].astype(int)

# Transform variable
df_road['speed_limit'] = np.log(df_road['speed_limit'])
scaler = MinMaxScaler(feature_range=(0,1))
df_road['speed_limit'] = scaler.fit_transform(df_road[['speed_limit']])

# Input data for prediction
input_row = df_road[:1]

# for debug, remove after testing
st.write(df_road[:1])
         
prediction = road_model.predict(input_row)
st.write(f"The chance of of road accident is :{prediction[0]}")
                                                          
                          
