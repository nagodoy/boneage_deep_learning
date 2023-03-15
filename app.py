import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
import random
import opencv-python as cv2

# Sidebar: Navigation
st.sidebar.header('Navigation')

page = st.sidebar.selectbox('Select',
                       ['Learn About Bone Age',
                        'Bone Age Predictor',
                        'Resources'])

# Education page
if page == 'Learn About Bone Age':
    st.title('About Bone Age')

    expander = st.beta_expander('What is bone age?')
    expander.write("'Bone age is an interpretation of skeletal maturity, typically based on radiographs of the left hand and wrist or knee, that has provided useful information in various clinical settings for over 75 years.' (Creo AL, Schwenk WF. Bone age: a handy tool for pediatric providers. Pediatrics. 2017;140(6).)")
    '\n'
    expander = st.beta_expander('What are current methods of determining bone age?')
    expander.write("""
                   * Tanner-White method: ...
                   * Greulich-Pyle method: ...
                   """)
    '\n'
    expander = st.beta_expander('What are some clinical uses for bone age?')
    expander.write("""
                   * Diagnosing certain growth (endocrinologic) conditions
                   * Determining which patients would benefit from treatment
                   * Monitoring treatment
                   * Predicting adult height
                   """)
    '\n'
    expander = st.beta_expander('What are some non-clinical uses for bone age?')
    expander.write("""
                   * Athletics
                   * Forensics
                   * Legal/policy
                   """)

# Bone age prediction
@st.cache
def get_model(model_path):
    model = load_model(model_path,
                       custom_objects={'mae_months': mae_months})
    return model

# Function for metric
def mae_months(y_true, y_pred):
    return mean_absolute_error((boneage_std*y_true + boneage_mean), (boneage_std*y_pred + boneage_mean))

# Function for processing image
def process_image(img, img_size=(299, 299)):
    # Code adapted from https://medium.com/swlh/building-a-deep-learning-flower-classifier-cfdbd59f0210
    image = ImageOps.fit(img, img_size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = preprocess_input(img)
    img_data = img_resize[np.newaxis,...]
    return img_data

# Function for returning prediction
def predict(model, img_data, gender):
    if gender == 'Female':
        gender_input = np.array([0])
    elif gender == 'Male':
        gender_input = np.array([1])    
    pred = mean + std*(model.predict([img_data, gender_input]))
    return pred

if page == 'Bone Age Predictor':
    
    st.title('Bone Age Prediction')
    
    # Load model
    model = get_model('attn_gender_model2.h5')
    
    # Mean & std dev for evaluation
    mean = 127.23862147637749
    std = 41.45201616171413

    # Upload image
    uploaded_file = st.file_uploader('Upload an image', type='png')
    if uploaded_file is not None:        
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded image', use_column_width=True)
        '\n'
        gender = st.radio('Sex:', ('Male', 'Female'))
        '\n'
        
        if st.button('Predict bone age'):
            img_data = process_image(img)
            boneage = predict(model, img_data, gender)[0][0] / 12
            boneage = round(boneage, 1)
            st.write(boneage, 'years')

# Resources page
if page == 'Resources':
    st.title('Resources')
