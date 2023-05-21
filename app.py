import streamlit as st
import pickle

# -------- Importing Model ----------
with open('SVC_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ----------- Title ---------
st.title('Breast Cancer Prediction App')
st.write('The breast cancer dataset is a collection of data that is used to train and test machine learning algorithms for breast cancer detection. The dataset contains 569 instances, each of which is described by 30 features. The features are measurements of the cell nuclei present in a digitized image of a fine needle aspirate (FNA) of a breast mass. The goal of the machine learning algorithm is to predict whether the instance is malignant (M) or benign (B).')

# ----------- Reset Function -------------
def session_state_reset():
    for key in st.session_state.keys():
        del st.session_state[key]

# --------- Widgets -----------
st.header('Input Form:')
with st.form(key='my_widgets'):
    radius = st.slider('Radius (millimeters)', min_value=0.0, max_value=50.0, step=0.01, value=0.0, key='radius')
    texture = st.slider('Texture (gray-scale values)', min_value=0.0, max_value=50.0, step=0.01, value=0.0, key='texture')
    smoothness = st.slider('Smoothness (1/(standard deviation of radius lengths))', min_value=0.0, max_value=1.0, step=0.001, value=0.0, key='smoothness')
    compactness = st.slider('Compactness (1 - (perimeter^2 / area))', min_value=0.0, max_value=1.0, step=0.001, value=0.0, key='compactness')
    concavity = st.slider('Concavity (1/(number of concave portions of the contour))', min_value=0.0, max_value=1.0, step=0.001, value=0.0, key='concavity')
    symmetry = st.slider('Symmetry', min_value=0.0, max_value=1.0, step=0.001, value=0.0, key='symmetry')
    fractal_dimension = st.slider('Fractal Dimension (Dimensionless)', min_value=0.0, max_value=1.0, step=0.001, value=0.0, key='fractal_dimension')
    col1, space1, col2, space2 = st.columns([10,0.1,9,50])
    with col1:
        predict = st.form_submit_button(label='Predict')
    with col2:
        reset = st.form_submit_button(label='Reset')

# ---------- Call the model -----------
prediction = ['','']
if predict:
    prediction = model.predict([[radius, texture, smoothness, compactness, concavity, symmetry, fractal_dimension]])

# ---------- Reset button -------------
if reset:
    session_state_reset()

# ----------- Display result ------------
st.header('Prediction Result:')
if prediction[0] == 0:
    st.write('The tumor is malignant.')
    st.write('A malignant breast tumor is a cancerous growth that can spread to other parts of the body. Malignant breast tumors are the most common type of cancer in women, and they can be life-threatening.')
elif prediction[0] == 1:
    st.write('The tumor is benign.')
    st.write('A benign breast tumor is a non-cancerous growth that does not spread to other parts of the body. Benign breast tumors are not usually life-threatening, and they are typically removed with surgery.')
else:
    st.write('The result will be shown here.')
