import streamlit as st
import pickle

# -------- Importing Model ----------
@st.cache_data
def load_model():
    with open('clf_sel.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

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
    radius = st.slider('Mean Radius (millimeters)', min_value=0.0, max_value=50.0, step=0.01, value=0.0, key='radius')
    texture = st.slider('Mean Texture (gray-scale values)', min_value=0.0, max_value=50.0, step=0.01, value=0.0, key='texture')
    smoothness = st.slider('Mean Smoothness (1/(standard deviation of radius lengths))', min_value=0.0, max_value=1.0, step=0.001, value=0.0, key='smoothness')
    compactness = st.slider('Mean Compactness (1 - (perimeter^2 / area))', min_value=0.0, max_value=1.0, step=0.001, value=0.0, key='compactness')
    texture_error = st.slider('Texture Error (gray-scale values)', min_value=0.0, max_value=50.0, step=0.01, value=0.0, key='texture_error')
    col1,space,col2 = st.columns([8,1,50])
    with col1:
        predict = st.form_submit_button(label='Predict')
    with col2:
        reset = st.form_submit_button(label='Reset')
# ---------- Call the model -----------
prediction = ['','']
if predict:
    prediction = model.predict([[radius, texture, smoothness, compactness, texture_error]])

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
