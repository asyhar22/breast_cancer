import streamlit as st
import pickle
import sklearn 

# -------- Importing Model ----------
with open('SVC_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ----------- Title ---------
st.title('Breast Cancer Prediction App')

# --------- Widgets -----------
radius = st.slider('Radius', 1, 100, 15)
texture = st.slider('Texture', 1, 100, 50)
smoothness = st.slider('Smoothness', 1, 100, 75)
compactness = st.slider('Compactness', 1, 100, 80)
concavity = st.slider('Concavity', 1, 100, 90)
symmetry = st.slider('Symmetry', 1, 100, 95)
fractal_dimension = st.slider('Fractal Dimension', 1, 100, 100)

# ---------- Call the model -----------
prediction = model.predict([[radius, texture, smoothness, compactness, concavity, symmetry, fractal_dimension]])

# ----------- Display result ------------
if prediction[0] == 0:
    st.write('The tumor is malignant.')
else:
    st.write('The tumor is benign.')
